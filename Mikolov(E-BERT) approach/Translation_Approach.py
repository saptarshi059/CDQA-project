#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle5 as pickle
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from tqdm import tqdm

UMLS_path = os.path.abspath('Train_KGE/UMLS_KG_MT+SN')

with open(os.path.join(UMLS_path, 'entity2idx.pkl'), 'rb') as file:
    entity2id = pickle.load(file)

entity_embeddings = pd.read_csv(os.path.join(UMLS_path, os.path.relpath('embeddings/distmult/ent_embedding.tsv')), sep='\t', header=None)

all_entities = set(entity2id.keys())

print('All Files Loaded...')

def common_terms_gen(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_vocab = set(tokenizer.get_vocab().keys())
    common_terms = list(all_entities.intersection(model_vocab))
    print(f"Number of KGE's common with {model_name} vocabulary: {len(common_terms)}")
    return common_terms

def data_gen(common_terms, model_name):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model_embeddings = model.get_input_embeddings()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_vocab = tokenizer.get_vocab()
    
    src = []
    tgt = []
    for term in tqdm(common_terms):
        src.append(entity_embeddings.iloc[entity2id[term]].to_numpy())
        tgt.append(model_embeddings(torch.LongTensor([model_vocab[term]])).detach().cpu().numpy())
    
    return src, tgt

def weight_matrix_compute(model_name):
    KGE_embeddings, Model_embeddings = data_gen(common_terms_gen(model_name), model_name)
    W = np.linalg.lstsq(np.vstack(KGE_embeddings),np.vstack(Model_embeddings), rcond=None)[0]
    return W

bert_W = weight_matrix_compute('phiyodr/bert-base-finetuned-squad2')
#scibert_W = weight_matrix_compute('ktrapeznikov/scibert_scivocab_uncased_squad_v2')

def homogenizer(weight_matrix, model_name):
    homogenized_embeddings = {}
    for entity_name, index in entity2id.items():
        homogenized_embeddings[entity_name] = torch.FloatTensor(np.matmul(weight_matrix.T, entity_embeddings.iloc[index].to_numpy()).reshape(1,-1))
    
    print(f'Saving Homogenized Embeddings for {model_name}...')
    pd.DataFrame(list(homogenized_embeddings.items()), columns = ['Entity', 'Embedding']).to_pickle(f'Mikolov_to_{model_name.replace("/","_")}.pkl')


homogenizer(bert_W, 'phiyodr/bert-base-finetuned-squad2')
#homogenizer(scibert_W, 'ktrapeznikov/scibert_scivocab_uncased_squad_v2')