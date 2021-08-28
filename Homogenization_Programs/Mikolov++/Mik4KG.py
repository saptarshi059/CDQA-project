#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle5 as pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel, logging
from tqdm import tqdm
import numpy as np
import torch

logging.set_verbosity(50)

with open('KGT.pkl', 'rb') as file:
    KGT = pickle.load(file)

with open('entity2idx.pkl', 'rb') as file:
    entity2id = pickle.load(file)

ent_embeddings = pd.read_csv('ent_embedding.tsv', sep='\t', header=None)
MRREL_rel2desc = pd.read_pickle('MRREL_rel2desc.pkl')
SEM_NW_rel2desc = pd.read_pickle('SEM_NW_rel2desc.pkl')
total_rel2desc = pd.concat([MRREL_rel2desc, SEM_NW_rel2desc], ignore_index=True).drop_duplicates()

model_name = 'phiyodr/bert-base-finetuned-squad2'

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Model loaded on device: {device}')

print('Necessary Files Loaded...')

using_pooler_output = True
src = []
tgt = []
with torch.no_grad():
    for triple in tqdm(KGT.itertuples()):
        natural_text = triple.E1 + ' ' + total_rel2desc.query('REL==@triple.Rel').Description.values[0] + ' ' + triple.E2
        inputs = tokenizer(natural_text, return_tensors='pt')
        inputs.to(device)
        output = model(**inputs)
        if using_pooler_output == True:
            src.append(ent_embeddings.iloc[entity2id[triple.E1]].to_numpy())
            tgt.append(output['pooler_output'].detach().cpu().numpy())
        else:
            src.append(ent_embeddings.iloc[entity2id[triple.E1]].to_numpy())
            tgt.append(output['last_hidden_state'][0][0].detach().cpu().numpy().reshape(1,-1))

weight_matrix = np.linalg.lstsq(np.vstack(src),np.vstack(tgt), rcond=None)[0]

homogenized_embeddings = {}
for entity_name, index in tqdm(entity2id.items()):
    homogenized_embeddings[entity_name] = torch.FloatTensor(np.matmul(weight_matrix.T, ent_embeddings.iloc[index].to_numpy()).reshape(1,-1))

print(f'Saving Homogenized Embeddings for {model_name}...')
pd.DataFrame(list(homogenized_embeddings.items()), columns = ['Entity', 'Embedding']).to_pickle(f'Mikolov_to_{model_name.replace("/","_")}.pkl')