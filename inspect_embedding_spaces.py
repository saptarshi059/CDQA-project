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
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from transformers import AdamW, AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer


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


def do_pca(x, n=2):
    pca_model = PCA(n_components=n)
    new_x = pca_model.fit_transform(x)

    return new_x


def do_tsne(x, perplexity, n_iter):
    tsne = TSNE(2, verbose=1, n_jobs=8, perplexity=perplexity, n_iter=n_iter)
    tsne_proj = tsne.fit_transform(x)

    return tsne_proj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dte_lookup_table_fp',
                        default='NN-DTE-to-phiyodr-bert-base-finetuned-squad2.pkl'
                        # default='Mikolov_to_phiyodr_bert-base-finetuned-squad2.pkl',
                        # default='DTE_to_ktrapeznikov_biobert_v1.1_pubmed_squad_v2.pkl',
                        # default='DTE_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl'
                        )
    parser.add_argument('--model_name',
                        # default='ktrapeznikov/scibert_scivocab_uncased_squad_v2',
                        # default='clagator/biobert_squad2_cased',
                        # default='navteca/roberta-base-squad2',
                        default='phiyodr/bert-base-finetuned-squad2',
                        # default='ktrapeznikov/biobert_v1.1_pubmed_squad_v2',
                        # default='ktrapeznikov/scibert_scivocab_uncased_squad_v2',
                        help='Type of model to use from HuggingFace')
    parser.add_argument('--out', default='out/embedding_space_inspection', help='Directory to put output')
    parser.add_argument('--tsne_perplexity', default=25, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    kge_name = args.dte_lookup_table_fp[:-4]
    outdir = os.path.join(args.out, kge_name)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plt_fp = os.path.join(outdir, 'embedding_space_viz_{}perp.png'.format(args.tsne_perplexity))

    DTE_Model_Lookup_Table = pickle.load(open(args.dte_lookup_table_fp, 'rb'))
    print('DTE_Model_Lookup_Table.columns: {}'.format(DTE_Model_Lookup_Table.columns))
    dtes = DTE_Model_Lookup_Table['UMLS_Embedding'].tolist()
    dtes = np.vstack(dtes)
    print('dtes.shape: {}'.format(dtes.shape))
    n_dte = dtes.shape[0]
    dte_labels = [1 for _ in range(dtes.shape[0])]

    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    initial_input_embeddings = model.get_input_embeddings().weight.detach().numpy()
    print('initial_input_embeddings.shape: {}'.format(initial_input_embeddings.shape))
    n_initial = initial_input_embeddings.shape[0]
    og_labels = [0 for _ in range(initial_input_embeddings.shape[0])]

    agg_embeddings = np.concatenate([initial_input_embeddings, dtes], axis=0)
    print('agg_embeddings.shape: {}'.format(agg_embeddings.shape))

    print('Performing pca...')
    agg_pca = do_pca(agg_embeddings, n=50)
    print('Performing TSNE...')
    agg_tsne = do_tsne(agg_pca, perplexity=args.tsne_perplexity, n_iter=1000)

    initial_tsne = agg_tsne[:n_initial]
    first_1000_tsne = initial_tsne[:999]
    # initial_tsne = initial_tsne[1000:]
    dte_tsne = agg_tsne[-n_dte:]
    print('initial_tsne: {}'.format(initial_tsne.shape))
    print('first_1000_tsne: {}'.format(first_1000_tsne.shape))
    print('dte_tsne: {}'.format(dte_tsne.shape))

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    # first_1000_ids = [i for i in range(999)]
    # first_1000_tokens = tokenizer.convert_ids_to_tokens(first_1000_ids)
    # for i, t in enumerate(first_1000_tokens):
    #     print('{}: {}'.format(i, t))
    # input('okty')
    cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    unk_id = tokenizer.convert_tokens_to_ids('[UNK]')
    mask_id = tokenizer.convert_tokens_to_ids('[MASK]')

    print('Plotting...')
    plt.scatter(initial_tsne[:, 0], initial_tsne[:, 1], c='green', label='Contextual')
    plt.scatter(dte_tsne[:, 0], dte_tsne[:, 1], c='red', label='DTE')
    plt.scatter(first_1000_tsne[:, 0], first_1000_tsne[:, 1], c='pink', label='[UNUSED]')

    plt.scatter(initial_tsne[cls_id, 0], initial_tsne[cls_id, 1], c='black', label='[CLS]')
    plt.scatter(initial_tsne[pad_id, 0], initial_tsne[pad_id, 1], c='orange', label='[PAD]')
    plt.scatter(initial_tsne[sep_id, 0], initial_tsne[sep_id, 1], c='blue', label='[SEP]')
    plt.scatter(initial_tsne[unk_id, 0], initial_tsne[unk_id, 1], c='purple', label='[UNK]')
    plt.scatter(initial_tsne[mask_id, 0], initial_tsne[mask_id, 1], c='yellow', label='[MASK]')

    # plt.scatter(initial_tsne[7592, 0], initial_tsne[7592, 1], c='brown', label='hello')

    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('{}\nEmbedding Space'.format(kge_name))
    plt.savefig(plt_fp, bbox_inches='tight')
    plt.clf()






