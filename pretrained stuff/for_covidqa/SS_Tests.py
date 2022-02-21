#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from numpy.random import randint
from tqdm.notebook import tqdm
import re

with open('NN-DTE-to-phiyodr-bert-base-finetuned-squad2.pkl', 'rb') as f:
    our_approach_df = pd.read_pickle(f)

with open('Mikolov_to_phiyodr_bert-base-finetuned-squad2.pkl', 'rb') as f:
    mik_approach_df = pd.read_pickle(f)
    
print('Necessary Files loaded...')

model = AutoModelForQuestionAnswering.from_pretrained('phiyodr/bert-base-finetuned-squad2')
model_embeddings = model.get_input_embeddings()
tokenizer = AutoTokenizer.from_pretrained('phiyodr/bert-base-finetuned-squad2')

print('Model and tokenizer loaded...')


# In[2]:


our_sim_scores = []
mik_sim_scores = []

cos = torch.nn.CosineSimilarity()
model_elements_subset = {}

count = 0
number_of_elements_to_select = tokenizer.__len__()

while count != number_of_elements_to_select:
    idx = randint(0, tokenizer.__len__())
    token = tokenizer._convert_id_to_token(idx)
    #We are trying to exclude numbers from the computation cause that doesn't really tell us anything
    if (not re.search('\d+', token)) and (token not in model_elements_subset.keys()):
        model_elements_subset[token] = model_embeddings(torch.LongTensor([idx]))
        count+= 1


# In[4]:


def compute_sim_stats(df, topk=20):
    sim_scores = {}
    for row in tqdm(df.itertuples(index=False)):
        for mod_el in model_elements_subset.items():
            sim_scores[(row.Entity, mod_el[0])] = cos(row.UMLS_Embedding.detach(), mod_el[1].detach()).item()
    scores = torch.FloatTensor(list(sim_scores.values()))
    print(f'Mean Sim: {torch.mean(scores)}')
    print(f'Std. Dev. Sim: {torch.std(scores)}')
    print(f'...\nTop {topk} similar pairs (Entity, Model)')
    print(sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)[0:topk])

compute_sim_stats(our_approach_df)
compute_sim_stats(mik_approach_df)

