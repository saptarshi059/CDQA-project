#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Reading in the necessary files
import pickle5 as pickle
import os
import pandas as pd
from transformers import AutoTokenizer

BERT_variant = 'phiyodr/bert-base-finetuned-squad2'
tokenizer = AutoTokenizer.from_pretrained(BERT_variant)

UMLS_KG_path = os.getcwd()

with open(os.path.join(UMLS_KG_path, 'entity2idx.pkl'), 'rb') as f:
    entity2id = pickle.load(f)

KGE_path = os.path.join(UMLS_KG_path, os.path.relpath('embeddings/distmult'))

ent_embeddings = pd.read_csv(os.path.join(KGE_path, 'ent_embedding.tsv'), sep='\t', header=None)

print('Loaded all necessary files...')


# In[ ]:


#Creating training dataset
import torch
from tqdm import tqdm
import numpy as np

'''
#Instead of storing the full target vector, I am storing the indices of the natural text word pieces. 
This allows us to create the target representation at runtime.
'''
entity_embeddings = []
multiple_hot_targets_indices = []

def gen_sample(entity_name, entity_index):
    return torch.FloatTensor(ent_embeddings.iloc[entity_index]), tokenizer(entity_name)['input_ids']
    
print('Creating training samples according to the conversion scheme...')
for entity_name, entity_index in tqdm(entity2id.items()):
    train, test = gen_sample(entity_name, entity_index)
    entity_embeddings.append(train)
    multiple_hot_targets_indices.append(test)

pd.DataFrame(zip(entity_embeddings, multiple_hot_targets_indices), columns=['train', 'test']).to_pickle('Entity_Homogenization_data.pkl')

print('FFN training dataset created...')