#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Loading trained model & setting it to eval mode
from MyNN import FFNN
from transformers import AutoTokenizer
import pandas as pd
import os
import torch
import pickle5 as pickle

BERT_variant = 'navteca/roberta-base-squad2'

UMLS_KG_path = os.getcwd()

KGE_path = os.path.join(UMLS_KG_path, os.path.relpath('embeddings/distmult'))

ent_embeddings = pd.read_csv(os.path.join(KGE_path, 'ent_embedding.tsv'), sep='\t', header=None)

with open('entity2idx.pkl', 'rb') as f:
    entity2id = pickle.load(f)

input_dimension = len(ent_embeddings.columns)
output_dimension = AutoTokenizer.from_pretrained(BERT_variant).vocab_size
number_of_hidden_layers = 10
hidden_dimension = 768

device = torch.device('cpu')

model = FFNN(input_dimension, output_dimension, number_of_hidden_layers, hidden_dimension)
model.load_state_dict(torch.load('Homogenizer.pt', map_location=device))
model.to(device)
model.eval()

print(f'Model loaded on device: {device} ...')


# In[ ]:


#Converting entity KGE's to target model equivalents
from tqdm import tqdm

entity_names = []
converted_embeddings = []

with torch.no_grad():
    for e_name, index in tqdm(entity2id.items()):
        entity_names.append(e_name)
        #Discarding softmax output
        _, homogenized_embedding = model(torch.Tensor(ent_embeddings.iloc[index].values,                                                       device=device).reshape(1,-1))
        converted_embeddings.append(homogenized_embedding)
        
print('Saving Homogenized Embeddings...')
pd.DataFrame(zip(entity_names, converted_embeddings),  columns = ['Entity', 'Embedding']).to_pickle(f"NN-DTE-to-{BERT_variant.replace('/','-')}.pkl")

