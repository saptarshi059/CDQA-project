__author__ = 'Connor Heaton and Saptarshi Sengupta'

import os
import math
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

import pickle5 as pickle
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_constant_schedule_with_warmup

from datasets import PubmedQADataset


class PubmedQARunner(object):
    def __init__(self, gpu, fold_dir, args):
        self.rank = gpu
        self.fold_dir = fold_dir
        self.args = args
        self.fold_no = int(fold_dir[-1])

        print('Initializing PubmedQARunner on device {}...'.format(gpu))
        if self.args.on_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.rank))
            torch.cuda.set_device(self.device)

        print('\ttorch.cuda.device_count(): {}'.format(torch.cuda.device_count()))
        dist.init_process_group('nccl',
                                world_size=len(self.args.gpus),
                                rank=self.rank)

        self.out = getattr(self.args, 'out', '../out')
        self.batch_size = getattr(self.args, 'batch_size', 40)
        self.n_epochs = getattr(self.args, 'n_epochs', 3)
        self.lr = getattr(self.args, 'l2', 5e-5)
        self.l2 = getattr(self.args, 'l2', 0.001)
        self.max_len = getattr(self.args, 'max_len', 512)
        self.model_name = getattr(self.args, 'model_name', 'navteca/roberta-base-squad2')
        self.use_kge = getattr(self.args, 'use_kge', False)
        self.use_dict = getattr(self.args, 'use_dict', False)
        self.concat_kge = getattr(self.args, 'concat_kge', False)
        self.fancy_concat = getattr(self.args, 'fancy_concat', False)
        self.random_kge = getattr(self.args, 'random_kge', False)
        self.n_warmup_iters = getattr(self.args, 'n_warmup_iters', 0)
        self.dte_lookup_table_fp = getattr(self.args, 'dte_lookup_table_fp', 0.1)
        self.on_cpu = getattr(self.args, 'on_cpu', False)

        pred_dir = os.path.join(self.out, 'preds')
        if self.rank == 0 and not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        self.pred_fp_tmplt = os.path.join(pred_dir, 'fold_{}_{}_preds_e{}.csv')

        print('PubmedQARunner on device {} creating datasets...'.format(gpu))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dtes = None
        if self.use_kge:
            DTE_Model_Lookup_Table = pickle.load(open(args.dte_lookup_table_fp, 'rb'))
            custom_domain_term_tokens = []
            domain_terms = DTE_Model_Lookup_Table['Entity'].tolist()
            custom_umls_tokens = ['[{}]'.format(dt) for dt in domain_terms]
            custom_dict_tokens = ['#{}#'.format(dt) for dt in domain_terms]
            if args.use_kge:
                custom_domain_term_tokens.extend(custom_umls_tokens)
            if args.use_dict:
                custom_domain_term_tokens.extend(custom_dict_tokens)

            self.tokenizer.add_tokens(custom_domain_term_tokens)

            dtes = []

            if args.use_kge:
                umls_dtes = DTE_Model_Lookup_Table['UMLS_Embedding'].tolist()
                dtes.extend(umls_dtes)
            if args.use_dict:
                dict_dtes = DTE_Model_Lookup_Table['Dictionary_Embedding'].tolist()
                dtes.extend(dict_dtes)

            if args.random_kge:
                print('Replacing DTEs with random tensors...')
                dtes = [torch.rand(1, 768) for _ in dtes]

            print('dtes[0]: {}'.format(dtes[0]))
            self.dtes = torch.cat(dtes, dim=0)  # .to(self.device)

        train_data_fp = os.path.join(self.fold_dir, 'train_set.json')
        self.dataset = PubmedQADataset(self.args, train_data_fp, self.tokenizer)
        if self.args.on_cpu:
            data_sampler = None
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                           num_replicas=args.world_size,
                                                                           rank=self.rank,
                                                                           shuffle=True,
                                                                           )
        self.data_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=self.args.n_data_workers, pin_memory=True, sampler=data_sampler)
        self.n_iters = int(math.ceil(len(self.dataset) / (self.args.batch_size * len(self.args.gpus))))

        dev_data_fp = os.path.join(self.fold_dir, 'dev_set.json')
        self.aux_dataset = PubmedQADataset(self.args, dev_data_fp, self.tokenizer)
        if self.args.on_cpu:
            data_sampler = None
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(self.aux_dataset,
                                                                           num_replicas=args.world_size,
                                                                           rank=self.rank,
                                                                           shuffle=False,
                                                                           )
        self.aux_data_loader = DataLoader(self.aux_dataset, batch_size=self.args.batch_size, shuffle=False,
                                          num_workers=self.args.n_data_workers, pin_memory=True, sampler=data_sampler)
        self.aux_n_iters = int(math.ceil(len(self.aux_dataset) / (self.args.batch_size * len(self.args.gpus))))

        print('PubmedQARunner on device {} creating model...'.format(gpu))
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
        if self.use_kge:
            initial_input_embeddings = self.model.get_input_embeddings().weight
            new_input_embedding_weights = torch.cat([initial_input_embeddings, self.dtes], dim=0)
            new_input_embeddings = nn.Embedding.from_pretrained(new_input_embedding_weights, freeze=False)
            self.model.set_input_embeddings(new_input_embeddings)

        self.model = self.model.to(self.device)

        if not self.args.on_cpu:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)

        self.summary_writer = None
        if self.rank == 0:
            self.summary_writer = SummaryWriter(log_dir=self.args.tb_dir)

        no_decay = ['layernorm', 'norm']
        param_optimizer = list(self.model.named_parameters())
        no_decay_parms = []
        reg_parms = []
        for n, p in param_optimizer:
            if any(nd in n for nd in no_decay):
                no_decay_parms.append(p)
            else:
                reg_parms.append(p)

        optimizer_grouped_parameters = [
            {'params': reg_parms, 'weight_decay': self.l2},
            {'params': no_decay_parms, 'weight_decay': 0.0},
        ]
        if self.rank == 0:
            print('n parms: {}'.format(len(param_optimizer)))
            print('len(optimizer_grouped_parameters[0]): {}'.format(len(optimizer_grouped_parameters[0]['params'])))
            print('len(optimizer_grouped_parameters[1]): {}'.format(len(optimizer_grouped_parameters[1]['params'])))
        self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.lr, betas=(0.9, 0.95))
        self.scheduler = None
        if self.n_warmup_iters > 0:
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                               num_warmup_steps=self.n_warmup_iters)

        self.run()

    def run(self):
        for epoch in range(self.n_epochs):
            if self.rank == 0:
                print('Performing epoch {} of {}'.format(epoch, self.n_epochs))

            self.model.train()
            self.run_one_epoch(epoch, 'train')

            if self.rank == 0:
                print('Saving model...')
                if not self.args.on_cpu:
                    torch.save(self.model.module.state_dict(),
                               os.path.join(self.args.model_save_dir, self.args.ckpt_file_tmplt.format(self.fold_no,
                                                                                                       epoch)))
                else:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.args.model_save_dir, self.args.ckpt_file_tmplt.format(self.fold_no,
                                                                                                       epoch)))
            dist.barrier()

            self.model.eval()
            with torch.no_grad():
                self.run_one_epoch(epoch, 'train-dev')

    def run_one_epoch(self, epoch, mode):
        if mode == 'train':
            dataset = self.data_loader
            n_iters = self.n_iters
        else:
            dataset = self.aux_data_loader
            n_iters = self.aux_n_iters

        write_lines = ['id,pred,label']
        iter_since_grad_accum = 1
        last_batch_end_time = None
        agg_correct_ps = []
        pred_list, label_list = [], []
        for batch_idx, batch_data in enumerate(dataset):
            global_item_idx = (epoch * n_iters) + batch_idx
            batch_start_time = time.time()

            input_ids = batch_data['input_ids'].to(self.device, non_blocking=True).squeeze(1)
            attention_mask = batch_data['attention_mask'].to(self.device, non_blocking=True).squeeze(1)
            token_type_ids = batch_data['token_type_ids'].to(self.device, non_blocking=True).squeeze(1)
            labels = batch_data['label'].to(self.device, non_blocking=True).squeeze(1)
            item_ids = batch_data['item_id'].to(self.device, non_blocking=True).squeeze(1)

            if epoch == 0 and batch_idx == 0 and self.rank == 0:
                print('input_ids: {}'.format(input_ids.shape))
                print('attention_mask: {}'.format(attention_mask.shape))
                print('token_type_ids: {}'.format(token_type_ids.shape))
                print('label: {}'.format(labels.shape))
                print('item_id: {}'.format(item_ids.shape))

            question_str = batch_data['question_str']
            context_str = batch_data['context_str']

            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                labels=labels
            )
            loss = output[0]
            logits = output[1]

            if mode == 'train':
                loss.backward()

            _, preds = torch.topk(logits, 1, dim=-1)

            preds = preds.detach()
            labels = labels.detach()

            # gather predictions from different devices
            if not self.on_cpu and self.args.world_size > 1:
                preds_list = [torch.zeros_like(preds) for _ in range(self.args.world_size)]
                labels_list = [torch.zeros_like(labels) for _ in range(self.args.world_size)]
                item_ids_list = [torch.zeros_like(item_ids) for _ in range(self.args.world_size)]

                dist.all_gather(preds_list, preds)
                dist.all_gather(labels_list, labels)
                dist.all_gather(item_ids_list, item_ids)

                preds = torch.cat(preds_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
                item_ids = torch.cat(item_ids_list, dim=0)

            preds = preds.view(-1)
            labels = labels.view(-1)
            item_ids = item_ids.view(-1)
            correct_ps = []
            batch_pred_list, batch_label_list = [], []

            for i in range(preds.shape[0]):
                this_line = '{},{},{}'.format(
                    item_ids[i].item(), preds[i].item(), labels[i].item()
                )
                write_lines.append(this_line)
                correct_ps.append(1 if preds[i].item() == labels[i].item() else 0)
                pred_list.append(preds[i].item())
                batch_pred_list.append(preds[i].item())
                label_list.append(labels[i].item())
                batch_label_list.append(labels[i].item())

            batch_acc = sum(correct_ps) / len(correct_ps)
            agg_correct_ps.extend(correct_ps)
            f1 = f1_score(batch_label_list, batch_pred_list, average='macro')

            if global_item_idx % self.args.grad_summary_every == 0 \
                    and self.summary_writer is not None and mode == 'train' \
                    and self.args.grad_summary and global_item_idx != 0:
                for name, p in self.model.named_parameters():
                    if p.grad is not None and p.grad.data is not None:
                        self.summary_writer.add_histogram('grad/{}'.format(name), p.grad.data,
                                                          (epoch * n_iters) + batch_idx)
                        self.summary_writer.add_histogram('weight/{}'.format(name), p.data,
                                                          (epoch * n_iters) + batch_idx)

            if global_item_idx % self.args.print_every == 0 and self.rank == 0:
                batch_elapsed_time = time.time() - batch_start_time
                if last_batch_end_time is not None:
                    time_btw_batches = batch_start_time - last_batch_end_time
                else:
                    time_btw_batches = 0.0

                print_str = 'Fold {0} {1} - epoch: {2}/{3} iter: {4}/{5} loss: {6:2.4f} Acc: {7:3.4f}% F1: {8:.3f}'.format(
                    self.fold_no, mode, epoch, self.n_epochs, batch_idx, n_iters, loss, batch_acc * 100, f1
                )
                print_str = '{0} Time: {1:.2f}s ({2:.2f}s)'.format(print_str, batch_elapsed_time, time_btw_batches)
                print(print_str)
                last_batch_end_time = time.time()

            if (global_item_idx % self.args.summary_every == 0 and self.summary_writer is not None) or (
                    mode == 'train-dev' and self.summary_writer is not None and global_item_idx % int(
                self.args.summary_every / 3) == 0):

                if loss is not None:
                    self.summary_writer.add_scalar('loss/{}'.format(mode), loss,
                                                   (epoch * n_iters) + batch_idx)
                    self.summary_writer.add_scalar('batch_acc/{}'.format(mode), batch_acc,
                                                   (epoch * n_iters) + batch_idx)
                    self.summary_writer.add_scalar('f1/{}'.format(mode), f1,
                                                   (epoch * n_iters) + batch_idx)

            if iter_since_grad_accum == self.args.n_grad_accum and mode == 'train':
                # print('OPTIMIZER STEP')
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                iter_since_grad_accum = 1
            else:
                iter_since_grad_accum += 1

        if self.rank == 0:
            agg_acc = sum(agg_correct_ps) / len(agg_correct_ps)
            agg_f1 = f1_score(label_list, pred_list, average='macro')
            print_str = '* Fold {0} Epoch {1} {2} Avg acc: {3:3.4f}% F1: {4:2.4f} *'.format(
                self.fold_no, epoch, mode, agg_acc * 100, agg_f1
            )
            print('*' * len(print_str))
            print(print_str)
            print('*' * len(print_str))

            # self.pred_fp_tmplt = os.path.join(pred_dir, 'fold_{}_{}_preds_e{}.csv')
            pred_fp = self.pred_fp_tmplt.format(self.fold_no, mode, epoch)
            with open(pred_fp, 'w+') as f:
                f.write('\n'.join(write_lines))






