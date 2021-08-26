__author__ = 'Connor Heaton and Saptarshi Sengupta'

import re
import gc
import os
import json
import math
import time
import torch
import string
import random
import argparse
import datetime
import collections

import numpy as np
import pandas as pd
import pickle5 as pickle

import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from transformers import get_constant_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, AutoTokenizer, AutoModelForQuestionAnswering

# from custom_input import custom_input_rep


class CovidQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) if key not in ['question_texts', 'context_texts'] else val[idx] for key, val
                in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def preprocess_input(dataset, tokenizer_, n_stride=64, max_len=512, n_neg=1):
    answers = dataset['answer'].to_list()
    context = dataset['context'].to_list()

    question_texts = dataset['question'].to_list()
    context_texts = dataset['context'].to_list()
    # print('len(question_texts): {}'.format(len(question_texts)))
    # print('len(context_texts): {}'.format(len(context_texts)))

    pad_on_right = tokenizer_.padding_side == "right"

    for answer, context in zip(dataset['answer'], dataset['context']):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters

    encodings = tokenizer_(
        dataset['question'].to_list() if pad_on_right else dataset['context'].to_list(),
        dataset['context'].to_list() if pad_on_right else dataset['question'].to_list(),
        truncation='longest_first',
        stride=n_stride,
        padding=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        max_length=max_len,
    )
    # print('encodings.keys(): {}'.format(encodings.keys()))
    # print('encodings[input_ids]: {}'.format(len(encodings['input_ids'])))

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = encodings.pop("overflow_to_sample_mapping")
    # print('sample_mapping: {}'.format(len(sample_mapping)))
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = encodings.pop("offset_mapping")
    # print('offset_mapping: {}'.format(len(offset_mapping)))

    # input('sample_mapping: {}'.format(sample_mapping))
    # input('offset_mapping: {}'.format(offset_mapping))

    encodings["start_positions"] = []
    encodings["end_positions"] = []

    positive_idxs = []
    neg_idxs_by_sample = {}

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = encodings["input_ids"][i]
        cls_index = input_ids.index(tokenizer_.cls_token_id)
        # print('offsets: {}'.format(offsets))
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = encodings.sequence_ids(i)
        # print('sequence_ids: {}'.format(sequence_ids))

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        # print('dataset[\'answer\']: {}'.format(dataset['answer']))
        # answers = dataset['answer'][sample_index]
        i_answers = answers[sample_index]
        # print('i_answers: {}'.format(i_answers))

        # If no answers are given, set the cls_index as answer.
        if i_answers["answer_start"] is None:
            # input('i_answers["answer_start"]: {}'.format(i_answers["answer_start"]))
            encodings["start_positions"].append(cls_index)
            encodings["end_positions"].append(cls_index)

            curr_neg_idxs_for_sample = neg_idxs_by_sample.get(sample_index, [])
            curr_neg_idxs_for_sample.append(i)
            neg_idxs_by_sample[sample_index] = curr_neg_idxs_for_sample
        else:

            # Start/end character index of the answer in the text.
            start_char = i_answers["answer_start"]
            end_char = start_char + len(i_answers["text"])
            # print('start_char: {}'.format(start_char))
            # print('end_char: {}'.format(end_char))

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            # print('** token_start_index: {} **'.format(token_start_index))

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            # print('** token_end_index: {} **'.format(token_end_index))

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                # print('offsets[token_start_index]: {}'.format(offsets[token_start_index]))
                # print('offsets[token_end_index]: {}'.format(offsets[token_end_index]))
                encodings["start_positions"].append(cls_index)
                encodings["end_positions"].append(cls_index)

                curr_neg_idxs_for_sample = neg_idxs_by_sample.get(sample_index, [])
                curr_neg_idxs_for_sample.append(i)
                neg_idxs_by_sample[sample_index] = curr_neg_idxs_for_sample
                # input('appending cls index')
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                encodings["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                encodings["end_positions"].append(token_end_index + 1)

                positive_idxs.append(i)

    # print('encodings.keys: {}'.format(encodings.keys()))
    # for k, v in encodings.items():
    #     if type(v) == list:
    #         print('k: {} v: {}'.format(k, len(v)))

    encodings["question_texts"] = []
    encodings["context_texts"] = []
    for i in range(len(encodings['input_ids'])):

        i_question_text = question_texts[sample_mapping[i]]
        i_input_ids = encodings['input_ids'][i]
        i_input_text = tokenizer_.decode(i_input_ids, skip_special_tokens=True)
        if i_input_text.startswith(i_question_text):
            i_context_text = i_input_text[len(i_question_text):]
        else:
            i_context_text = i_input_text[:-len(i_question_text)]

        encodings["question_texts"].append(i_question_text)
        encodings["context_texts"].append(i_context_text)

    for k, v in encodings.items():
        print('k: {} v: {}'.format(k, len(v)))

    print('len(positive_idxs): {}'.format(len(positive_idxs)))
    print('Selecting up to {} negative records for each sample...'.format(n_neg))

    for sample_idx, potential_neg_idxs in neg_idxs_by_sample.items():
        if n_neg > 0:
            selected_neg_idxs = random.choices(potential_neg_idxs, k=n_neg)
        else:
            selected_neg_idxs = potential_neg_idxs
        positive_idxs.extend(selected_neg_idxs)

    positive_idxs = list(sorted(positive_idxs))
    print('len(positive_idxs): {}'.format(len(positive_idxs)))

    for encoding_key in encodings.keys():
        encodings[encoding_key] = [encodings[encoding_key][i] for i in positive_idxs]

    for k, v in encodings.items():
        print('k: {} v: {}'.format(k, len(v)))

    # input('okty')

    return encodings


class DistributedFoldTrainer(object):
    def __init__(self, gpu, arg_d):
        self.rank = gpu

        self.model_ckpt_fp = arg_d['model_ckpt_fp']
        self.tb_dir = arg_d['tb_dir']
        self.dataset = arg_d['dataset']
        self.train_idxs = arg_d['train_ids']
        self.model_name = arg_d['model_name']
        self.n_stride = arg_d['N_STRIDE']
        self.max_len = arg_d['MAX_LEN']
        self.world_size = arg_d['world_size']
        self.batch_size = arg_d['batch_size']
        self.lr = arg_d['lr']
        self.n_epochs = arg_d['n_epochs']
        self.use_kge = arg_d['USE_KGE']
        self.fold = arg_d['fold']
        self.n_splits = arg_d['n_splits']
        self.n_neg_records = arg_d['n_neg_records']
        self.dtes = arg_d['dtes']
        self.warmup_proportion = arg_d['warmup_proportion']
        self.seed = arg_d['seed']
        self.concat_kge = arg_d['concat_kge']
        self.my_maker = arg_d['my_maker']
        self.args = arg_d['args']

        self.device = torch.device('cuda:{}'.format(self.rank)) if torch.cuda.is_available() else torch.device('cpu')
        torch.cuda.set_device(self.device)
        print('Device for GPU {}: {}'.format(self.rank, self.device))

        torch.manual_seed(self.seed)
        dist.init_process_group('nccl',
                                world_size=self.world_size,
                                rank=self.rank)

        print('Creating tokenizer and dataset on device {}...'.format(self.rank))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.dataset = CovidQADataset(preprocess_input(self.dataset.iloc[self.train_idxs], self.tokenizer,
        #                                                n_stride=self.n_stride, max_len=self.max_len,
        #                                                n_neg=self.n_neg_records))
        self.fold_n_iters = int(len(self.dataset) / (self.batch_size * self.world_size))
        data_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                       num_replicas=self.world_size,
                                                                       rank=self.rank, shuffle=True)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                      pin_memory=True, sampler=data_sampler)

        print('Creating model on device {}...'.format(self.rank))
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.dtes = self.dtes.to(self.model.device)

        if self.use_kge:
            initial_input_embeddings = self.model.get_input_embeddings().weight

            if self.rank == 0:
                print('initial_input_embeddings.device: {}'.format(initial_input_embeddings.device))
                print('dtes.device: {}'.format(self.dtes.device))

                print('initial_input_embeddings: {}'.format(initial_input_embeddings.shape))
                print('dtes: {}'.format(self.dtes.shape))
            new_input_embedding_weights = torch.cat([initial_input_embeddings, self.dtes], dim=0)
            if self.rank == 0:
                print('new_input_embedding_weights: {}'.format(new_input_embedding_weights.shape))

            new_input_embeddings = nn.Embedding.from_pretrained(new_input_embedding_weights, freeze=False)
            self.model.set_input_embeddings(new_input_embeddings)

        self.model.train()
        self.model = DDP(self.model, device_ids=self.args.gpus)

        print('Creating optimizer on device {}...'.format(self.rank))
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
            {'params': reg_parms, 'weight_decay': 0.01},
            {'params': no_decay_parms, 'weight_decay': 0.0},
        ]

        self.n_iters = int(math.ceil(len(self.dataset) / (self.batch_size * self.world_size)))
        self.optim = AdamW(optimizer_grouped_parameters, lr=self.lr)
        self.scheduler = None
        if self.warmup_proportion > 0.0:
            n_warmup_iters = int(
                len(self.dataset) * self.n_epochs * self.warmup_proportion / (self.batch_size * self.world_size))
            if self.rank == 0:
                print('** n_warmup_iters: {} **'.format(n_warmup_iters))
            self.scheduler = get_constant_schedule_with_warmup(self.optim,
                                                               num_warmup_steps=n_warmup_iters)

        self.summary_writer = None
        # if self.rank == 0:
        #     print('** GPU 0 creating summary writer **')
        #     self.summary_writer = SummaryWriter(log_dir=self.tb_dir)

        dist.barrier()
        print('GPU {} passed barrier... running...'.format(self.rank))
        self.run()

    def run(self):
        for epoch in range(self.n_epochs):
            if self.rank == 0:
                print('Performing epoch {} of {}'.format(epoch, self.n_epochs))

            self.run_one_epoch(epoch)

        if self.rank == 0:
            print('Training done, saving model...')
            self.model.eval()
            torch.save(self.model.module.state_dict(), self.model_ckpt_fp)
        dist.barrier()

    def run_one_epoch(self, epoch):
        n_orig_token_counts, n_dte_hit_counts = [], []

        for batch_idx, batch in enumerate(self.data_loader):
            # if batch_idx > 2:
            #     break
            batch_start_time = time.time()
            # input('batch.keys(): {}'.format(batch.keys()))

            # question_texts = batch['question_texts']
            # context_texts = batch['context_texts']
            input_ids = batch['input_ids'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)

            if batch_idx == 0 and self.rank == 0:
                print('GPU {} input_ids - shape: {} device: {}'.format(self.rank, input_ids.shape, input_ids.device))
                print('GPU {} token_type_ids - shape: {} device: {}'.format(self.rank, token_type_ids.shape, token_type_ids.device))
                print('GPU {} attention_mask - shape: {} device: {}'.format(self.rank, attention_mask.shape, attention_mask.device))
                print('GPU {} start_positions - shape: {} device: {}'.format(self.rank, start_positions.shape, start_positions.device))
                print('GPU {} end_positions - shape: {} device: {}'.format(self.rank, end_positions.shape, start_positions.device))

            outputs = self.model(inputs_embeds=None, input_ids=input_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 start_positions=start_positions, end_positions=end_positions)

            loss = outputs[0]
            loss.backward()

            if self.rank == 0:
                batch_elapsed_time = time.time() - batch_start_time
                print_str = 'Fold: {6}/{7} Epoch: {0}/{1} Iter: {2}/{3} Loss: {4:.4f} Time: {5:.2f}s'
                print_str = print_str.format(epoch, self.n_epochs,
                                             batch_idx, self.fold_n_iters,
                                             loss, batch_elapsed_time,
                                             self.fold, self.n_splits)
                print(print_str)

            self.optim.step()
            self.optim.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            # print('Worker {} finished batch {}...'.format(self.rank, batch_idx))

            dist.barrier()

        avg_n_hits = sum(n_dte_hit_counts) / len(n_dte_hit_counts) if len(n_dte_hit_counts) > 0 else 0.0
        pct_replaced = [x / y if y is not 0 else 0.0 for x, y in zip(n_orig_token_counts, n_dte_hit_counts)]
        avg_pct_replaced = sum(pct_replaced) / len(pct_replaced) if len(pct_replaced) > 0 else 0.0


