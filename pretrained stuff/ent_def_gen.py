#python ent_def_gen.py -f for_pubmedqa/scibert/UMLS_Only_NN-DTE-to-gsarti-scibert-nli.pkl 

import sqlite3
import pickle
from tqdm import tqdm
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_name', required=True)
args = parser.parse_args()

with open(os.path.abspath(args.file_name), 'rb') as file:
    s = pickle.load(file)

conn = sqlite3.connect('../umls.db')
cursor = conn.cursor()

entity = []
definition = []
UMLS_Embedding = []
c = 0
for row in tqdm(s.itertuples(index=False)):
    cursor.execute('''SELECT DEF FROM MRDEF WHERE CUI = '%s' ''' % row.CUI)
    results = cursor.fetchall()
    if results == []:
        c += 1
    else:
        entity.append(row.Entity)
        UMLS_Embedding.append(row.UMLS_Embedding)
        definition.append(results[0][0])
    
print(f"Number of entities with no definition: {c}")            
pd.DataFrame(zip(entity, UMLS_Embedding, definition), columns=['Entity', 'UMLS_Embedding' ,'Definition']).to_pickle('Entity_Definition.pkl')
conn.close()