#!/usr/bin/env python
# coding: utf-8

#Reading in the necessary files
import pickle5 as pickle
import os
import pandas as pd
from transformers import AutoTokenizer

BERT_variant = 'navteca/roberta-base-squad2'
tokenizer = AutoTokenizer.from_pretrained(BERT_variant)

UMLS_KG_path = os.getcwd()

with open(os.path.join(UMLS_KG_path, 'KGT.pkl'), 'rb') as f:
    KGT = pickle.load(f)
    
with open(os.path.join(UMLS_KG_path, 'entity2idx.pkl'), 'rb') as f:
    entity2id = pickle.load(f)

with open(os.path.join(UMLS_KG_path, 'relation2idx.pkl'), 'rb') as f:
    relation2id = pickle.load(f)

KGE_path = os.path.join(UMLS_KG_path, os.path.relpath('embeddings/distmult'))

ent_embeddings = pd.read_csv(os.path.join(KGE_path, 'ent_embedding.tsv'), sep='\t', header=None)
rel_embeddings = pd.read_csv(os.path.join(KGE_path, 'rel_embedding.tsv'), sep='\t', header=None)    

MRREL_rel2desc = pd.read_pickle('MRREL_rel2desc.pkl')
SEM_NW_rel2desc = pd.read_pickle('SEM_NW_rel2desc.pkl')

total_rel2desc = pd.concat([MRREL_rel2desc, SEM_NW_rel2desc], ignore_index=True)

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
mean_embeddings = []
multiple_hot_targets_indices = []

def gen_sample(triple):
    
    '''
    We can expand using this scheme since we've taken care of the correct direction during KGT construction.
    It will, always be E1 - REL - E2.
    '''
    natural_text = triple.E1 + ' ' +                     total_rel2desc.query('REL==@triple.Rel').Description.values[0] + ' ' + triple.E2

    '''
    #Creating the target multiple-hot vector.
    target = np.zeros(vocab_size)

    #Replacing those elements in the target vector with 1, which are activated for this sample.
    np.put(target, tokenizer(natural_text)['input_ids'], 1)
    '''
        
    #Creating the mean embedding for the triple
    E1_tensor = torch.from_numpy(ent_embeddings.iloc[entity2id[triple.E1]].to_numpy()).float()
    Rel_tensor = torch.from_numpy(rel_embeddings.iloc[relation2id[triple.Rel]].to_numpy()).float()
    E2_tensor = torch.from_numpy(ent_embeddings.iloc[entity2id[triple.E2]].to_numpy()).float()

    return torch.mean(torch.stack([E1_tensor, Rel_tensor, E2_tensor]), dim=0), tokenizer(natural_text)['input_ids']
    
print('Creating training samples according to the conversion scheme...')
for trple in tqdm(KGT.itertuples()):
    train, test = gen_sample(trple)
    mean_embeddings.append(train)
    multiple_hot_targets_indices.append(test)

pd.DataFrame(zip(mean_embeddings, multiple_hot_targets_indices), columns=['train', 'test']).to_pickle('Homogenization_data.pkl')

print('FFN training dataset created...')