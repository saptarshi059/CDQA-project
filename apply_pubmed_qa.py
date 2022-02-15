__author__ = 'Connor Heaton and Saptarshi Sengupta'


import os
import math
import torch
import argparse
import datetime

import pandas as pd
from torch.utils.data import DataLoader

from argparse import Namespace
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import PubmedQADataset


def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("huh?")


def autoconvert(s):
    if s in ['[BOS]', '[EOS]']:
        return s
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass

    if s[0] == '[' and s[-1] == ']':
        s = s[1:-1]
        s = [ss.strip().strip('\'') for ss in s.split(',')]

    return s


def read_model_args(fp):
    m_args = {}

    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line == '':
                arg, val = line.split('=')
                arg = arg.strip()
                val = val.strip()

                val = autoconvert(val)
                m_args[arg] = val

    m_args = Namespace(**m_args)

    return m_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='/home/czh/nvme1/pubmedqa/data/test_set.json')

    parser.add_argument('--model_id', default='20220210-170238')
    parser.add_argument('--epoch', default=6, type=int)
    parser.add_argument('--batch_size', default=96, type=int)

    parser.add_argument('--path_to_models', default='out')
    parser.add_argument('--subpath_to_model_ckpts', default='models/')

    args = parser.parse_args()

    model_dir = os.path.join(args.path_to_models, args.model_id)
    model_ckpts_dir = os.path.join(model_dir, args.subpath_to_model_ckpts)
    model_args_fp = os.path.join(model_dir, 'args.txt')
    model_args = read_model_args(model_args_fp)
    model_name = model_args.model_name

    ckpts_to_apply = [
        fp for fp in os.listdir(model_ckpts_dir) if fp.endswith('{}e.pt'.format(args.epoch))
    ]
    ckpts_to_apply = list(sorted(ckpts_to_apply, key=lambda x: int(x[4])))
    # print('ckpts_to_apply: {}'.format(ckpts_to_apply))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = PubmedQADataset(model_args, args.data, tokenizer)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=3, pin_memory=True)
    n_iters = int(math.ceil(len(dataset) / args.batch_size))

    agg_stats = {'acc': [], 'f1': []}
    for fold_ckpt in ckpts_to_apply:
        fold_no = int(fold_ckpt[4])
        ckpt_fp = os.path.join(model_ckpts_dir, fold_ckpt)

        print('Applying ckpt from fold {} epoch {}...'.format(fold_no, args.epoch))
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        map_location = {'cuda:0': 'cpu'}
        state_dict = torch.load(ckpt_fp, map_location=map_location)
        model.load_state_dict(state_dict, strict=True)
        model = model.to('cuda:0')
        model.eval()

        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                print('\tProcessing batch {}/{}...'.format(batch_idx, n_iters))
                input_ids = batch_data['input_ids'].to('cuda:0', non_blocking=True).squeeze(1)
                attention_mask = batch_data['attention_mask'].to('cuda:0', non_blocking=True).squeeze(1)
                token_type_ids = batch_data['token_type_ids'].to('cuda:0', non_blocking=True).squeeze(1)
                labels = batch_data['label'].to('cuda:0', non_blocking=True).squeeze(1)
                item_ids = batch_data['item_id'].to('cuda:0', non_blocking=True).squeeze(1)

                output = model(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    labels=labels
                )
                logits = output[1]
                _, preds = torch.topk(logits, 1, dim=-1)
                preds = preds.view(-1)
                labels = labels.view(-1)

                for i in range(preds.shape[0]):
                    fold_preds.append(preds[i].item())
                    fold_labels.append(labels[i].item())

        fold_matches = [1 if p == l else 0 for p, l in zip(fold_preds, fold_labels)]
        fold_acc = sum(fold_matches) / len(fold_matches)
        fold_f1 = f1_score(fold_preds, fold_labels, average='macro')
        print('\tAcc: {0:3.4f}% F1: {1:2.4f}'.format(fold_acc * 100, fold_f1))

        agg_stats['acc'].append(fold_acc)
        agg_stats['f1'].append(fold_f1)

    agg_acc = sum(agg_stats['acc']) / len(agg_stats['acc'])
    agg_f1 = sum(agg_stats['f1']) / len(agg_stats['f1'])
    print('agg_acc: {0:3.4f} agg_f1: {1:3.4f}'.format(agg_acc, agg_f1))
    with open(os.path.join(model_dir, 'test_stats_e{}.txt'.format(args.epoch)), 'w+') as f:
        f.write('agg_acc: {0:3.4f} agg_f1: {1:3.4f}'.format(agg_acc, agg_f1))


