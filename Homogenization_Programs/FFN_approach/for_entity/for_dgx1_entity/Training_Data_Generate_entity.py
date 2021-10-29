#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reading in the necessary files
import pickle5 as pickle
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel, logging
import torch

logging.set_verbosity(50)

BERT_variant = 'phiyodr/bert-base-finetuned-squad2'
tokenizer = AutoTokenizer.from_pretrained(BERT_variant)
model = AutoModel.from_pretrained(BERT_variant)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Model loaded on device: {device}')

model_embeddings = model.get_input_embeddings()

UMLS_KG_path = os.path.abspath('../Train_KGE/UMLS_KG_MT-original/')

with open(os.path.join(UMLS_KG_path, 'entity2idx.pkl'), 'rb') as f:
    entity2id = pickle.load(f)

ent_embeddings = pd.read_csv(os.path.join(UMLS_KG_path, os.path.relpath('embeddings/transe/ent_embedding.tsv')), sep='\t', header=None)

print('Loaded all necessary files...')


# In[2]:


#Creating training dataset - with subword embedding scheme
from tqdm import tqdm
import numpy as np

src = []
tgt = []
for ent_name, ent_index in tqdm(entity2id.items()):
    entity_tokens = tokenizer(ent_name, return_tensors='pt')['input_ids'][0]
    sw_embds = []
    for index in range(1, len(entity_tokens)-1):
        sw_embds.append(model_embeddings(entity_tokens[index].to(device)))
    src.append(ent_embeddings.iloc[ent_index].to_numpy())
    tgt.append(torch.mean(torch.vstack(sw_embds), dim=0).cpu().detach().numpy())
    
pd.DataFrame(zip(src, tgt), columns=['train', 'test']).to_pickle('Entity_Homogenization_data.pkl')

print('FFN training dataset created...')