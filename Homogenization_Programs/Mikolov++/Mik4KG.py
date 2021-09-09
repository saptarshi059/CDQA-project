#!/usr/bin/env python
# coding: utf-8

#python Mik4KG.py --UMLS_Path ../../../Train_KGE/UMLS_KG_MT+SN --BERT_Variant navteca/roberta-base-squad2 --KGE_Variant transe --THROUGH True --TRIPLES False --HS False --PO True

import pickle5 as pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel, logging
from tqdm import tqdm
import numpy as np
import torch
import os
import argparse

logging.set_verbosity(50)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--UMLS_Path', type=str)
parser.add_argument('--BERT_Variant', type=str)
parser.add_argument('--THROUGH', type=str2bool)
parser.add_argument('--TRIPLES', type=str2bool)
parser.add_argument('--HS', type=str2bool)
parser.add_argument('--PO', type=str2bool)
parser.add_argument('--KGE_Variant', type=str)
args = parser.parse_args()

if args.TRIPLES == True:
    MRREL_rel2desc = pd.read_pickle('MRREL_rel2desc.pkl')
    SEM_NW_rel2desc = pd.read_pickle('SEM_NW_rel2desc.pkl')
    total_rel2desc = pd.concat([MRREL_rel2desc, SEM_NW_rel2desc], ignore_index=True).drop_duplicates()

#Mapping b/w entity and corresponding ID
with open(os.path.join(os.path.abspath(args.UMLS_Path), 'entity2idx.pkl'), 'rb') as f:
    entity2id = pickle.load(f)

#Reading KGT dataframe
with open(os.path.join(os.path.abspath(args.UMLS_Path), 'KGT.pkl'), 'rb') as f:
    KGT = pickle.load(f)

ent_embeddings = pd.read_csv(os.path.join(os.path.abspath(args.UMLS_Path), os.path.relpath(f'embeddings/{args.KGE_Variant}/ent_embedding.tsv')), sep='\t', header=None)

model_name = args.BERT_Variant
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_embeddings = model.get_input_embeddings()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
model.to(device)
print(f'Model loaded on device: {device}')

print('Necessary Files Loaded...')

src = []
tgt = []

if args.THROUGH == False:
    for ent_name, ent_index in tqdm(entity2id.items()):
        entity_tokens = tokenizer(ent_name, return_tensors='pt')['input_ids'][0]
        sw_embds = []
        for index in range(1, len(entity_tokens)-1):
            sw_embds.append(model_embeddings(entity_tokens[index]))
        src.append(ent_embeddings.iloc[ent_index].to_numpy())
        tgt.append(torch.mean(torch.vstack(sw_embds), dim=0).detach().numpy())
else:
    number_of_layers = 4
    with torch.no_grad():
        if args.TRIPLES == True:
            for triple in tqdm(KGT.itertuples()):
                natural_text = triple.E1 + ' ' + total_rel2desc.query('REL==@triple.Rel').Description.values[0] + ' ' + triple.E2
                inputs = tokenizer(natural_text, return_tensors='pt')
                inputs.to(device)
                output = model(**inputs)
                if args.PO == True:
                    src.append(ent_embeddings.iloc[entity2id[triple.E1]].to_numpy())
                    tgt.append(output['pooler_output'].detach().cpu().numpy().reshape(1,-1))
                else:#[CLS] output
                    src.append(ent_embeddings.iloc[entity2id[triple.E1]].to_numpy())
                    tgt.append(output['last_hidden_state'][0][0].detach().cpu().numpy().reshape(1,-1))
        else:
            for ent_name, ent_index in tqdm(entity2id.items()):
                inputs = tokenizer(ent_name, return_tensors='pt')
                inputs.to(device)
                output = model(**inputs, output_hidden_states=True)
                if args.HS == True:
                    entity_tokens = output['hidden_states'][0][0].shape[0] - 2
                    vecs = []
                    for layer_idx in range(number_of_layers):
                        vecs.append([output['hidden_states'][layer_idx][0][i] for i in range(1,  entity_tokens + 1)])
                    c = torch.vstack([torch.hstack(vecs[x]).reshape(1, entity_tokens, 768) for x in range(number_of_layers)])
                    src.append(ent_embeddings.iloc[ent_index].to_numpy())
                    tgt.append(torch.mean(torch.mean(c, dim=0), dim=0).detach().cpu().numpy().reshape(1,-1))
                else:          
                    if args.PO == True:
                        src.append(ent_embeddings.iloc[ent_index].to_numpy())
                        tgt.append(output['pooler_output'].detach().cpu().numpy().reshape(1,-1))
                    else: #[CLS] output
                        src.append(ent_embeddings.iloc[ent_index].to_numpy())
                        tgt.append(output['last_hidden_state'][0][0].detach().cpu().numpy().reshape(1,-1))

weight_matrix = np.linalg.lstsq(np.vstack(src),np.vstack(tgt), rcond=None)[0]
homogenized_embeddings = {}
for entity_name, index in tqdm(entity2id.items()):
    homogenized_embeddings[entity_name] = torch.FloatTensor(np.matmul(weight_matrix.T, ent_embeddings.iloc[index].to_numpy()).reshape(1,-1))

print(f'Saving Homogenized Embeddings for {model_name}...')
pd.DataFrame(list(homogenized_embeddings.items()), columns = ['Entity', 'Embedding']).to_pickle(f'Mikolov++_to_{model_name.replace("/","_")}.pkl')