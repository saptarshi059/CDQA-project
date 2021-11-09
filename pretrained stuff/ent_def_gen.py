import sqlite3
import pandas as pd
from tqdm import tqdm

s = pd.read_pickle('NN-DTE-to-phiyodr-bert-base-finetuned-squad2.pkl')
conn = sqlite3.connect('umls.db')
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
pd.DataFrame(zip(entity, UMLS_Embedding, definition), columns=['Entity', 'UMLS_Embedding' ,'Definition']).to_csv('Entity_Definition.csv', index=False)
conn.close()