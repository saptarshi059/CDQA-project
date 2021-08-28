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

BERT_variant = 'phiyodr/bert-base-finetuned-squad2'

#UMLS_KGE_path = os.getcwd()

#ent_embeddings = pd.read_csv(os.path.join(UMLS_KGE_path, os.path.relpath('embeddings/distmult/ent_embedding.tsv')), sep='\t', header=None)

UMLS_KG_path = os.path.abspath('../Train_KGE/UMLS_KG/')

with open(os.path.join(UMLS_KG_path, 'entity2idx.pkl'), 'rb') as f:
    entity2id = pickle.load(f)

ent_embeddings = pd.read_csv(os.path.join(UMLS_KG_path, os.path.relpath('embeddings/distmult/ent_embedding.tsv')), sep='\t', header=None)

with open('Entity_Homogenization_data.pkl', 'rb') as f:
    dataset = pickle.load(f)

#input_dimension = len(ent_embeddings.columns)
#output_dimension = AutoTokenizer.from_pretrained(BERT_variant).vocab_size
input_dimension = dataset['train'].iloc[0].shape[0]
output_dimension = dataset['test'].iloc[0].shape[0]

number_of_hidden_layers = 5
hidden_dimension = 768

device = torch.device('cpu')

model = FFNN(input_dimension, output_dimension, number_of_hidden_layers, hidden_dimension)
model.load_state_dict(torch.load('Homogenizer.pt', map_location=device))
model.to(device)
model.eval()

print(f'Model loaded on device: {device} ...')

#Converting entity KGE's to target model equivalents
from tqdm import tqdm

entity_names = []
converted_embeddings = []

with torch.no_grad():
    for e_name, index in tqdm(entity2id.items()):
        entity_names.append(e_name)
        ent_embed = torch.FloatTensor(ent_embeddings.iloc[index].values).reshape(1,-1)
        #Discarding softmax output
        _, homogenized_embedding = model(ent_embed.to(device))
        converted_embeddings.append(homogenized_embedding)
        
print('Saving Homogenized Embeddings...')
pd.DataFrame(zip(entity_names, converted_embeddings), columns = ['Entity', 'Embedding']).to_pickle(f"NN-DTE-to-{BERT_variant.replace('/','-')}.pkl")

