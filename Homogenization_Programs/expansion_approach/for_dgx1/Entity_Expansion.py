#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Converting KGE to BERT embeddings (Domain Term Encoding (DTE)) - part1 (generating associated triples)
#[Entity Expansion]

import numpy as np
import os
import pandas as pd
import pickle5 as pickle
from tqdm import tqdm

#Mapping b/w entity and corresponding ID
with open('entity2idx.pkl', 'rb') as f:
    entity2id = pickle.load(f)

#Mapping b/w relation and corresponding ID
with open('relation2idx.pkl', 'rb') as f:
    relation2id = pickle.load(f)

#Reading KGT dataframe
with open('KGT.pkl', 'rb') as f:
    KGT = pickle.load(f)

def triple_gen(current_entity):
    triples = set()
    
    #Outgoing Relations
    results = KGT.query("E1==@current_entity")
    connected_entities = results.E2.to_list()
    outgoing_relations = results.Rel.to_list()
    for i in range(len(results)):
        triples.add((current_entity, outgoing_relations[i], connected_entities[i]))
    
    #Incoming Relations
    results = KGT.query("E2==@current_entity")
    connected_entities = results.E1.to_list()
    incoming_relations = results.Rel.to_list()
    for i in range(len(results)):
        triples.add((connected_entities[i], incoming_relations[i], current_entity))
    
    return [y for x in list(triples) for y in x]

triple_list = {}
for entity in tqdm(entity2id.keys()):
    triple_list[entity] = triple_gen(entity)


# In[ ]:


#Converting KGE to BERT embeddings (Domain Term Encoding (DTE)) - part2 (each KG item -> (KG item, KGE))
#KGE located here

ent_embeddings = pd.read_csv(os.path.abspath('embeddings/distmult/ent_embedding.tsv'), sep='\t', header=None)
rel_embeddings = pd.read_csv(os.path.abspath('embeddings/distmult/rel_embedding.tsv'), sep='\t', header=None)

'''
Associating each item in the triple list with respective embeddings. 
This is done to create an easy Domain Term BERT embedding matrix.
'''
for TL in tqdm(triple_list.values()):
    if TL == []:
        continue
    i = 0
    for index, item in enumerate(TL):
        if (i%3 == 0) or (i%3 == 2): #This item is an entity
            TL[index] = (item, ent_embeddings.iloc[entity2id[item]].to_numpy())
        else: #This item is a relation
            TL[index] = (item, rel_embeddings.iloc[relation2id[item]].to_numpy())
        i += 1
        
#Saving the expanded entities to disk
with open('expanded_entities.pkl', 'wb') as filehandle:
    pickle.dump(triple_list, filehandle)

print('Triple list for all entities saved...')