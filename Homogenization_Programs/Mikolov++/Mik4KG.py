#!/usr/bin/env python
# coding: utf-8

#python Mik4KG.py --UMLS_Path ../../../Train_KGE/UMLS_KG_MT+SN --BERT_Variant navteca/roberta-base-squad2

import pickle5 as pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel, logging
from tqdm import tqdm
import numpy as np
import torch
import os
import argparse

logging.set_verbosity(50)

parser = argparse.ArgumentParser()
parser.add_argument('--UMLS_Path', type=str)
parser.add_argument('--BERT_Variant', type=str)
args = parser.parse_args()

#Mapping b/w entity and corresponding ID
with open(os.path.join(os.path.abspath(args.UMLS_Path), 'entity2idx.pkl'), 'rb') as f:
    entity2id = pickle.load(f)

#Reading KGT dataframe
with open(os.path.join(os.path.abspath(args.UMLS_Path), 'KGT.pkl'), 'rb') as f:
    KGT = pickle.load(f)

ent_embeddings = pd.read_csv(os.path.join(os.path.abspath(args.UMLS_Path), os.path.relpath('embeddings/distmult/ent_embedding.tsv')), sep='\t', header=None)

model_name = args.BERT_Variant
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_embeddings = model.get_input_embeddings()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Model loaded on device: {device}')

print('Necessary Files Loaded...')

src = []
tgt = []
for ent_name, ent_index in tqdm(entity2id.items()):
    entity_tokens = tokenizer(ent_name, return_tensors='pt')['input_ids'][0]
    sw_embds = []
    for index in range(1, len(entity_tokens)-1):
        sw_embds.append(model_embeddings(entity_tokens[index]))
    src.append(ent_embeddings.iloc[ent_index].to_numpy())
    tgt.append(torch.mean(torch.vstack(sw_embds), dim=0).detach().numpy())

weight_matrix = np.linalg.lstsq(np.vstack(src),np.vstack(tgt), rcond=None)[0]

homogenized_embeddings = {}
for entity_name, index in tqdm(entity2id.items()):
    homogenized_embeddings[entity_name] = torch.FloatTensor(np.matmul(weight_matrix.T, ent_embeddings.iloc[index].to_numpy()).reshape(1,-1))

print(f'Saving Homogenized Embeddings for {model_name}...')
pd.DataFrame(list(homogenized_embeddings.items()), columns = ['Entity', 'Embedding']).to_pickle(f'Mikolov++_to_{model_name.replace("/","_")}.pkl')