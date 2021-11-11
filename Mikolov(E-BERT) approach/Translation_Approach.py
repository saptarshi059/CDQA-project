#!/usr/bin/env python
# coding: utf-8

#python Translation_Approach.py --BERT_Variant phiyodr/bert-base-finetuned-squad2

import pandas as pd
import pickle5 as pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--BERT_Variant', type=str)
args = parser.parse_args()

embeddings = pd.read_csv('embeddings.csv',header=None)
their_CUI_PC = pd.read_csv('their_CUI_PC.csv')

embeddings.rename(columns={0:'CUI'}, inplace=True)
print('Necessary files loaded...')

CUI_PC_Emb = their_CUI_PC.merge(embeddings, on='CUI', how='inner')

embs = []
for row in tqdm(CUI_PC_Emb.itertuples(index=False)):
    embs.append(np.array(list(row[2:])))

CUI_PC_Emb['Embedding'] = embs
CUI_PC_Emb.drop(columns=range(1,51), inplace=True)

all_entities = set(CUI_PC_Emb.PC.to_list())

tokenizer = AutoTokenizer.from_pretrained(args.BERT_Variant)
model_vocab = set(tokenizer.get_vocab().keys())
common_terms = list(all_entities.intersection(model_vocab))
print(f"Number of KGE's common with {args.BERT_Variant} vocabulary: {len(common_terms)}")

#For faster access
CUI_PC_Emb.set_index('PC', inplace=True)

model = AutoModelForQuestionAnswering.from_pretrained(args.BERT_Variant)
model_embeddings = model.get_input_embeddings()
model_vocab = tokenizer.get_vocab()

KGE_embeddings = []
Model_embeddings = []
for term in tqdm(common_terms):
    KGE_embeddings.append(CUI_PC_Emb.loc[term].Embedding)
    Model_embeddings.append(model_embeddings(torch.LongTensor([model_vocab[term]])).detach().cpu().numpy())

Weight_Matrix = np.linalg.lstsq(np.vstack(KGE_embeddings),np.vstack(Model_embeddings), rcond=None)[0]

homogenized_embeddings = {}
for term in tqdm(common_terms):
    homogenized_embeddings[term] = torch.FloatTensor(np.matmul(Weight_Matrix.T, CUI_PC_Emb.loc[term].Embedding).reshape(1,-1))

print(f'Saving Homogenized Embeddings for {model_name}...')
pd.DataFrame(list(homogenized_embeddings.items()), columns = ['Entity', 'UMLS_Embedding']).to_pickle(f'Mikolov_to_{model_name.replace("/","_")}.pkl')