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

from transformers import AdamW, AutoTokenizer, AutoModelForQuestionAnswering


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
                        default='Mikolov++_to_phiyodr_bert-base-finetuned-squad2.pkl'
                        # default='DTE_to_phiyodr_bert-base-finetuned-squad2.pkl',
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

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    kge_name = args.dte_lookup_table_fp[:-4]
    outdir = os.path.join(args.out, kge_name)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plt_fp = os.path.join(outdir, 'embedding_space_viz.png')

    DTE_Model_Lookup_Table = pickle.load(open(args.dte_lookup_table_fp, 'rb'))
    dtes = DTE_Model_Lookup_Table['Embedding'].tolist()
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
    agg_tsne = do_tsne(agg_pca, perplexity=75, n_iter=1000)

    initial_tsne = agg_tsne[:n_initial]
    dte_tsne = agg_tsne[-n_dte:]
    print('initial_tsne: {}'.format(initial_tsne.shape))
    print('dte_tsne: {}'.format(dte_tsne.shape))

    print('Plotting...')
    plt.scatter(initial_tsne[:, 0], initial_tsne[:, 1], c='green', label='OG')
    plt.scatter(dte_tsne[:, 0], dte_tsne[:, 1], c='red', label='DTE')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('{}\nEmbedding Space'.format(kge_name))
    plt.savefig(plt_fp, bbox_inches='tight')
    plt.clf()






