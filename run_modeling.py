__author__ = 'Connor Heaton and Saptarshi Sengupta'

import re
import gc
import os
import json
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
from transformers import get_constant_schedule_with_warmup
from transformers import AdamW, AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
# from custom_question_rep import custom_question_rep_gen
from custom_input import custom_input_rep
from custom_qa_pipeline import CustomQuestionAnsweringPipeline


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_covidqa(fp):
    # with open(fp, 'rb') as f:
    #     covidqa_dict = json.load(f)

    with open(fp, encoding='utf-8', errors='ignore') as json_data:
        covidqa_dict = json.load(json_data, strict=False)

    contexts = []
    questions = []
    answers = []
    for group in covidqa_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


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

        # print('i: {}'.format(i))
        # print('\ti_input_text: {}'.format(i_input_text))
        # print('\ti_question_text: {}'.format(i_question_text))
        # print('\ti_context_text: {}'.format(i_context_text))
        # print('\tstart_positions: {}'.format(encodings['start_positions'][i]))
        # print('\tend_positions: {}'.format(encodings['end_positions'][i]))
        # input('okty')

    # input('okty')
    # encodings.update({'question_texts': question_texts, 'context_texts': context_texts})
    # u_start_vals, u_start_cnts = np.unique(encodings['start_positions'], return_counts=True)
    # u_end_vals, u_end_cnts = np.unique(encodings['end_positions'], return_counts=True)

    # print('Start positions')
    # for v, c in zip(u_start_vals, u_start_cnts):
    #     print('v: {} count: {}'.format(v, c))
    #
    # print('End positions')
    # for v, c in zip(u_end_vals, u_end_cnts):
    #     print('v: {} count: {}'.format(v, c))

    for k, v in encodings.items():
        print('k: {} v: {}'.format(k, len(v)))

    print('len(positive_idxs): {}'.format(len(positive_idxs)))
    print('Selecting up to {} negative records for each sample...'.format(n_neg))
    for sample_idx, potential_neg_idxs in neg_idxs_by_sample.items():
        selected_neg_idxs = random.choices(potential_neg_idxs, k=n_neg)
        positive_idxs.extend(selected_neg_idxs)

    positive_idxs = list(sorted(positive_idxs))
    print('len(positive_idxs): {}'.format(len(positive_idxs)))

    for encoding_key in encodings.keys():
        encodings[encoding_key] = [encodings[encoding_key][i] for i in positive_idxs]

    for k, v in encodings.items():
        print('k: {} v: {}'.format(k, len(v)))

    # input('okty')

    return encodings


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_EM(df):
    EM = []
    for i in range(len(df)):
        a_gold = df['true_answer'][i]
        a_pred = df['predicted_answer'][i]
        EM.append(int(normalize_answer(a_gold) == normalize_answer(a_pred)))
    return np.mean(EM)


def compute_f1_main(df):
    F1 = []
    for i in range(len(df)):
        a_gold = df['true_answer'][i]
        a_pred = df['predicted_answer'][i]
        F1.append(compute_f1(a_gold, a_pred))
    return np.mean(F1)


class CovidQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) if key not in ['question_texts', 'context_texts'] else val[idx] for key, val
                in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def train_fold_distributed(rank, out_fp, dataset, train_idxs, model_name, n_stride, max_len, world_size, batch_size,
                           lr, n_epochs, use_kge, fold, n_splits, n_neg_records, dtes,
                           warmup_proportion=0.0, seed=16, l2=0.01):
    dist.init_process_group('nccl',
                            world_size=world_size,
                            rank=rank)
    torch.manual_seed(seed)

    device_ = torch.device('cuda:{}'.format(rank)) if torch.cuda.is_available() else torch.device('cpu')

    print('Creating tokenizer and dataset on device {}...'.format(rank))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = CovidQADataset(preprocess_input(dataset.iloc[train_idxs], tokenizer,
                                              n_stride=n_stride, max_len=max_len, n_neg=n_neg_records))
    fold_n_iters = int(len(dataset) / (batch_size * world_size))
    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                   num_replicas=world_size,
                                                                   rank=rank, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=True, sampler=data_sampler)

    print('Creating model on device {}...'.format(rank))
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model = model.to(device_)
    dtes = dtes.to(model.device)

    if use_kge:
        initial_input_embeddings = model.get_input_embeddings().weight

        print('initial_input_embeddings.device: {}'.format(initial_input_embeddings.device))
        print('dtes.device: {}'.format(dtes.device))

        print('initial_input_embeddings: {}'.format(initial_input_embeddings.shape))
        print('dtes: {}'.format(dtes.shape))
        new_input_embedding_weights = torch.cat([initial_input_embeddings, dtes], dim=0)
        print('new_input_embedding_weights: {}'.format(new_input_embedding_weights.shape))

        new_input_embeddings = nn.Embedding.from_pretrained(new_input_embedding_weights, freeze=False)
        model.set_input_embeddings(new_input_embeddings)

    dist.barrier()

    n_orig_token_counts, n_dte_hit_counts = [], []

    model.train()
    no_decay = ['layernorm', 'norm']
    param_optimizer = list(model.named_parameters())

    no_decay_parms = []
    reg_parms = []
    for n, p in param_optimizer:
        if any(nd in n for nd in no_decay):
            no_decay_parms.append(p)
        else:
            reg_parms.append(p)

    optimizer_grouped_parameters = [
        {'params': reg_parms, 'weight_decay': l2},
        {'params': no_decay_parms, 'weight_decay': 0.0},
    ]

    optim = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = None
    if warmup_proportion > 0.0:
        n_warmup_iters = int(len(dataset) * n_epochs * warmup_proportion / (batch_size * world_size))
        print('** n_warmup_iters: {} **'.format(n_warmup_iters))
        scheduler = get_constant_schedule_with_warmup(optim,
                                                      num_warmup_steps=n_warmup_iters)

    for epoch_idx in range(n_epochs):
        for batch_idx, batch in enumerate(data_loader):
            # if batch_idx > 2:
            #     break
            batch_start_time = time.time()
            optim.zero_grad()
            question_texts = batch['question_texts']
            context_texts = batch['context_texts']
            input_ids = batch['input_ids'].to(device_)
            attention_mask = batch['attention_mask'].to(device_)
            start_positions = batch['start_positions'].to(device_)
            end_positions = batch['end_positions'].to(device_)

            if use_kge:
                input_embds, offsets, attn_masks, input_ids = [], [], [], []
                for q_text, c_text in zip(question_texts, context_texts):
                    # re.sub(' +', ' ', q_text) this was returning a string, I guess if you assigned it to q_text, it would have worked.
                    with torch.no_grad():
                        # print('** KGE **')
                        custom_input_info = custom_input_rep(q_text, c_text)
                        this_input_embds, this_n_token_adj, this_attention_mask, _, in_ids, n_orig_tokens, n_dte_hits = custom_input_info
                        # print('this_n_token_adj: {}'.format(this_n_token_adj))
                        this_n_token_adj = torch.tensor([this_n_token_adj])
                        input_embds.append(this_input_embds.unsqueeze(0))
                        attn_masks.append(this_attention_mask.unsqueeze(0))
                        input_ids.append(in_ids.unsqueeze(0))
                        offsets.append(this_n_token_adj)
                        n_orig_token_counts.append(n_orig_tokens)
                        n_dte_hit_counts.append(n_dte_hits)

                input_embds = torch.cat(input_embds, dim=0).to(device_)
                input_ids = torch.cat(input_ids, dim=0).to(device_)
                offsets = torch.cat(offsets, dim=0).to(device_)

                # print('custom input_ids: {}'.format(input_ids.shape))
                # print('custom input_embds: {}'.format(input_embds.shape))

                attention_mask = torch.cat(attn_masks, dim=0).to(device_)

                start_positions = start_positions - offsets
                end_positions = end_positions - offsets
            else:
                # model_embds = model.get_input_embeddings()
                # input_embds = model_embds(input_ids)
                pass

            outputs = model(inputs_embeds=None, input_ids=input_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optim.step()
            if scheduler is not None:
                if batch_idx == 0 and epoch_idx == 0:
                    print('* scheduler.step() *')   # just to know it 'took'
                scheduler.step()

            batch_elapsed_time = time.time() - batch_start_time
            print_str = 'Fold: {6}/{7} Epoch: {0}/{1} Iter: {2}/{3} Loss: {4:.4f} Time: {5:.2f}s'
            print_str = print_str.format(epoch_idx, n_epochs,
                                         batch_idx, fold_n_iters,
                                         loss, batch_elapsed_time,
                                         fold, n_splits)
            if rank == 0:
                print(print_str)
    # only save once

    avg_n_hits = sum(n_dte_hit_counts) / len(n_dte_hit_counts)
    pct_replaced = [x / y if y is not 0 else 0.0 for x, y in zip(n_orig_token_counts, n_dte_hit_counts)]
    avg_pct_replaced = sum(pct_replaced) / len(pct_replaced)

    if use_kge:
        print_str = '** GPU {0} Avg n hits: {1:.2f} Avg replace pct: {2:.2f}'.format(rank,
                                                                                     avg_n_hits,
                                                                                     avg_pct_replaced)
        print(print_str)

    if rank == 0:
        print('Saving model...')
        model.eval()
        torch.save(model.state_dict(), out_fp)
    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        default='data/COVID-QA_cleaned.json',
                        # default='data/200423_covidQA.json',
                        help='Filepath to CovidQA dataset')
    parser.add_argument('--out', default='out', help='Directory to put output')

    parser.add_argument('--n_splits', default=5, help='How many folds to use for cross val', type=int)
    parser.add_argument('--batch_size', default=4, help='How many items to process as once', type=int)
    parser.add_argument('--lr', default=5e-5, help='How many items to process as once', type=float)
    parser.add_argument('--n_epochs', default=3, help='If training/fine-tuning, how many epochs to perform', type=int)
    parser.add_argument('--n_stride', default=164, help='How many folds to use for cross val', type=int)
    parser.add_argument('--max_len', default=512, help='How many folds to use for cross val', type=int)
    parser.add_argument('--model_name',
                        # default='ktrapeznikov/scibert_scivocab_uncased_squad_v2',
                        # default='clagator/biobert_squad2_cased',
                        # default='navteca/roberta-base-squad2',
                        default='deepset/roberta-base-squad2',
                        help='Type of model to use from HuggingFace')

    parser.add_argument('--use_kge', default=False, help='If KGEs should be place in input',
                        type=str2bool)
    parser.add_argument('--seed', default=16, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, help='Fuck Timo Moller', type=float)

    parser.add_argument('--dte_lookup_table_fp', default='DTE-to-navteca-roberta-base-squad2.pkl')
    parser.add_argument('--n_neg_records', default=1, type=int)

    parser.add_argument('--gpus', default=[0], help='Which GPUs to use', type=int, nargs='+')
    parser.add_argument('--port', default='12345', help='Port to use for DDP')

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    torch.manual_seed(args.seed)
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print('*' * len('* Model time ID: {} *'.format(curr_time)))
    print('* Model time ID: {} *'.format(curr_time))
    print('*' * len('* Model time ID: {} *'.format(curr_time)))

    model_outdir = os.path.join(args.out, curr_time)
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)

    model_ckpt_dir = os.path.join(model_outdir, 'models')
    if not os.path.exists(model_ckpt_dir):
        os.makedirs(model_ckpt_dir)

    model_ckpt_tmplt = os.path.join(model_ckpt_dir, 'model_fold{}.pt')

    model_arg_fp = os.path.join(model_outdir, 'args.txt')
    args_d = vars(args)
    with open(model_arg_fp, 'w+') as f:
        for k, v in args_d.items():
            f.write('{} = {}\n'.format(k, v))

    USE_KGE = args.use_kge
    effective_model_name = args.model_name.replace('/', '-')
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_out_fname = '{}{}_{}.txt'.format(effective_model_name,
                                           '_kge' if USE_KGE else '',
                                           curr_time)
    model_out_fp = os.path.join(model_outdir, model_out_fname)
    print('*** model_out_fp: {} ***'.format(model_out_fp))

    DTE_Model_Lookup_Table = pickle.load(open(args.dte_lookup_table_fp, 'rb'))
    dtes = DTE_Model_Lookup_Table['Embedding'].tolist()
    dtes = torch.cat(dtes, dim=0).to('cpu')
    # input('dtes: {}'.format(dtes.shape))

    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    all_contexts, all_questions, all_answers = read_covidqa(args.data)
    # Converting to a dataframe for easy k-fold splits
    full_dataset = pd.DataFrame(list(zip(all_contexts, all_questions, all_answers)),
                                columns=['context', 'question', 'answer'])


    # sample_encoded_inputs = tokenizer(text='the first string of text',
    #                                   text_pair='the second string of text')
    # input('sample_encoded_inputs: {}'.format(sample_encoded_inputs))

    N_STRIDE = args.n_stride
    # MAX_LEN = tokenizer.model_max_length if tokenizer.model_max_length <= 512 else 512
    MAX_LEN = args.max_len
    # some of the tokenizers return 1000000000000000019884624838656 as model_max_length for some reason

    fold_f1_score = []
    fold_EM_score = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
        print('FOLD {}'.format(fold))
        print('--------------------------------')
        model_ckpt_fp = model_ckpt_tmplt.format(fold)
        dtes = dtes.to('cpu')

        print('Training {} distributed model(s) for fold {}...'.format(len(args.gpus), fold))
        mp.spawn(train_fold_distributed, nprocs=len(args.gpus), args=(model_ckpt_fp, full_dataset, train_ids,
                                                                      args.model_name, N_STRIDE, MAX_LEN,
                                                                      len(args.gpus), args.batch_size,
                                                                      args.lr, args.n_epochs, USE_KGE,
                                                                      fold, args.n_splits, args.n_neg_records,
                                                                      dtes,
                                                                      args.warmup_proportion,
                                                                      args.seed))

        # Process is complete.
        print('Training process has finished...')

        print('Loading trained model ckpt...')
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        model = model.to(device)
        dtes = dtes.to(model.device)

        if USE_KGE:
            initial_input_embeddings = model.get_input_embeddings().weight
            new_input_embedding_weights = torch.cat([initial_input_embeddings, dtes], dim=0)
            new_input_embeddings = nn.Embedding.from_pretrained(new_input_embedding_weights, freeze=False)
            model.set_input_embeddings(new_input_embeddings)

        map_location = {'cuda:0': 'cuda:0'}
        state_dict = torch.load(model_ckpt_fp, map_location=map_location)
        model.load_state_dict(state_dict)

        # Print about testing
        print('Starting testing')

        # Evaluationfor this fold
        test_data = full_dataset.iloc[test_ids]
        # nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=-1 if device == torch.device('cpu') \
        #     else 0)

        # if USE_KGE:
        #     print('$$$ Using Custom QA Pipeline $$$')
        #     nlp = CustomQuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=-1 if device == torch.device('cpu') else 0)
        # else:
        #     print('$$$ Using Vanilla QA Pipeline $$$')
        #     nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=-1 if device == torch.device('cpu') else 0)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        nlp = CustomQuestionAnsweringPipeline(model=model, tokenizer=tokenizer,
                                              device=-1 if device == torch.device('cpu') else 0)

        with torch.no_grad():
            questions = []
            true_answers = []
            predicted_answers = []
            final_df = pd.DataFrame(columns=['question', 'true_answer', 'predicted_answer'])

            # Iterate over the test data and generate predictions
            for i in range(len(test_data)):
                # if i > 2:
                #     break
                context = test_data.iloc[i]['context']
                questions.append(test_data.iloc[i]['question'])
                true_answers.append(test_data.iloc[i]['answer']['text'])

                # Generate outputs
                QA_input = {'question': questions[i], 'context': context}
                input_embds = None
                attn_mask = None
                input_ids = None
                if USE_KGE:
                    # input_embds, offsets, attn_masks = [], [], []
                    with torch.no_grad():
                        # print('** KGE **')
                        custom_input_data = custom_input_rep(questions[i], context, max_length=args.max_len)
                        this_input_embds, this_n_token_adj, this_attention_mask, new_q_text, input_ids, _, _ = custom_input_data
                        # print('this_n_token_adj: {}'.format(this_n_token_adj))
                        this_n_token_adj = torch.tensor([this_n_token_adj])
                        input_embds = this_input_embds.unsqueeze(0)
                        attn_mask = this_attention_mask.unsqueeze(0)
                        input_ids = input_ids.unsqueeze(0)
                        offsets = this_n_token_adj
                        QA_input['question'] = new_q_text

                    # attention_mask = torch.cat(attn_masks, dim=0).to(device)
                else:
                    encoded_inputs = tokenizer(text=questions[i],
                                               text_pair=context,
                                               max_length=args.max_len,
                                               padding='longest',
                                               stride=N_STRIDE)
                    input_ids = encoded_inputs['input_ids']
                    attn_mask = encoded_inputs['attention_mask']

                # else:
                #     predicted_answer = nlp(QA_input)['answer']
                # print('input_embds: {}'.format(input_embds.shape))
                # print('attn_mask: {}'.format(attn_mask.shape))
                predicted_answer = nlp(
                    QA_input,
                    _input_ids_=input_ids,
                    _input_embds_=None,
                    _attention_mask_=attn_mask,
                    max_seq_len=MAX_LEN,
                    doc_stride=N_STRIDE
                )['answer']

                predicted_answers.append(predicted_answer)

            final_df['question'] = questions
            final_df['true_answer'] = true_answers
            final_df['predicted_answer'] = predicted_answers

        # Print F1
        fold_f1_score.append(compute_f1_main(final_df))
        print('F1 for fold {}: {}'.format(fold, fold_f1_score[fold]))

        # Print EM
        fold_EM_score.append(compute_EM(final_df))
        print('EM for fold {}: {}'.format(fold, fold_EM_score[fold]))

        del model
        del nlp
        del final_df
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("Avg. F1: {}".format(np.mean(fold_f1_score)))
    print("Avg. EM: {}".format(np.mean(fold_EM_score)))

    print('Writing results to file...')
    write_lines = ['Fold: {0}\tF1: {1:.4f}\tEM: {2:.4}'.format(f_idx, f1, em)
                   for f_idx, (f1, em) in enumerate(zip(fold_f1_score, fold_EM_score))]
    write_lines.append('Avg. F1: {}'.format(np.mean(fold_f1_score)))
    write_lines.append('Avg. EM: {}'.format(np.mean(fold_EM_score)))

    with open(model_out_fp, 'w+') as f:
        f.write('\n'.join(write_lines))
