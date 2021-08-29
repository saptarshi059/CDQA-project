#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Converting KGE to BERT embeddings (Domain Term Encoding (DTE)) - part3 (Passing KGE's through BERT)
#[Creating DTE Lookup Table]

from transformers import AutoModel, AutoTokenizer
import torch
import pickle5 as pickle
from tqdm import tqdm
import pandas as pd

#Loading triple_list
with open('expanded_entities.pkl', 'rb') as f:
    triple_list = pickle.load(f)

#if use_average == True, homogenized embedding is formed by averaging occurrences of the entity o/p
#Else, is == pooled model output.
def Create_DTE_BERT_LookUp_Table(model_name, use_average=True):
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    model_embeddings = model.get_input_embeddings()

    CLS_embedding = model_embeddings(torch.LongTensor([tokenizer.cls_token_id]).to(device))
    SEP_embedding = model_embeddings(torch.LongTensor([tokenizer.sep_token_id]).to(device))

    DTE_BERT_Matrix = {}

    with torch.no_grad():
        for key_entity, expansion in tqdm(triple_list.items()):
            expansion_embeddings = torch.FloatTensor([x[1] for x in expansion]).to(device)
            
            outputs = model(inputs_embeds = torch.unsqueeze(                                            torch.cat(                                            (CLS_embedding, expansion_embeddings, SEP_embedding)), dim = 1))
            
            if use_average == True:
                #Collecting all the embeddings for the current domain term in e[]
                e = []

                #Adding 1 to index to account for [CLS]
                entity_indices = [index + 1 for index, tup in enumerate(expansion) if tup[0] == key_entity]

                for i in entity_indices:
                    e.append(outputs[0][i])

                '''
                The BERT embedding for each entity will be the average of all its occurrences.
                *e provides all the elements of e (unpacking).
                '''
                DTE_BERT_Matrix[key_entity] = torch.mean(torch.stack([*e], dim = 0), dim = 0)
            else:
                DTE_BERT_Matrix[key_entity] = torch.unsqueeze(outputs[1], dim = 1)

    DTE_BERT_Lookup_Table = pd.DataFrame(list(DTE_BERT_Matrix.items()), columns = ['Entity', 'Embedding'])
    DTE_BERT_Matrix.clear()
    
    print('Saving converted embeddings...')
    DTE_BERT_Lookup_Table.to_pickle(f'Expansion-DTE_to_{model_name.replace("/","_")}.pkl')

Create_DTE_BERT_LookUp_Table('phiyodr/bert-base-finetuned-squad2')
#Create_DTE_BERT_LookUp_Table('ktrapeznikov/scibert_scivocab_uncased_squad_v2')

