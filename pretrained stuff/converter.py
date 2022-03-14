#!/usr/bin/env python
# coding: utf-8

#python converter.py -d test_data_Homogenization_data.pkl -b navteca/roberta-base-squad2

#Loading trained model & setting it to eval mode
from MyNN import FFNN
from transformers import AutoTokenizer
import pandas as pd
import os
import torch
import pickle5 as pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-hl', '--num_hidden_layers', default=1, type=int)
parser.add_argument('-hd', '--hidden_dim', default=300, type=int)
parser.add_argument('-d', '--data_file', required=True)
parser.add_argument('-b', '--bert_variant', default='phiyodr/bert-base-finetuned-squad2')

args = parser.parse_args()

BERT_variant = args.bert_variant

dataset = pickle.load(open(args.data_file, 'rb'))
input_dimension = dataset['train'].iloc[0].shape[0]
output_dimension = dataset['test'].iloc[0].shape[0]

number_of_hidden_layers = args.num_hidden_layers
hidden_dimension = args.hidden_dim

model = FFNN(input_dimension, output_dimension, number_of_hidden_layers, hidden_dimension)
model.load_state_dict(torch.load('Homogenizer.pt'))
model.eval()

print('Model loaded...')

criterion = torch.nn.MSELoss() #To measure test loss

#Converting entity KGE's to target model equivalents
from tqdm import tqdm

entity_names = []
converted_embeddings = []
CUI = []
loss_vals = []

with torch.no_grad():
    for row in tqdm(dataset.itertuples(index=False)):
        CUI.append(row.CUI)
        entity_names.append(row.PC)
        ent_embed = torch.FloatTensor(row.train).reshape(1,-1)
        _, homogenized_embedding = model(ent_embed)
        converted_embeddings.append(homogenized_embedding)

        loss_vals.append(criterion(torch.FloatTensor(homogenized_embedding).reshape(1,-1), torch.FloatTensor(row.test).reshape(1,-1)).item())

print('Saving Homogenized Embeddings...')
pd.DataFrame(zip(CUI, entity_names, converted_embeddings), columns = ['CUI', 'Entity', 'UMLS_Embedding']).to_pickle(f"UMLS_Only_NN-DTE-to-{BERT_variant.replace('/','-')}.pkl")

print(f'Average Test Loss: {np.mean(loss_vals)}')