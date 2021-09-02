#python Mik4KG.py --UMLS_Path ../../../Train_KGE/UMLS_KG_MT+SN

import pickle5 as pickle
import pandas as pd
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--UMLS_Path', type=str)
args = parser.parse_args()

ent_embeddings = pd.read_csv(os.path.join(os.path.abspath(args.UMLS_Path), os.path.relpath('embeddings/distmult/ent_embedding.tsv')), sep='\t', header=None)

with open(os.path.join(os.path.abspath(args.UMLS_Path), 'entity2idx.pkl'), 'rb') as f:
    entity2id = pickle.load(f)

homogenized_embeddings = {}
for entity_name, index in tqdm(entity2id.items()):
    homogenized_embeddings[entity_name] = torch.FloatTensor(ent_embeddings.iloc[index].to_numpy()).reshape(1,-1)

print(f'Saving Homogenized Embeddings for {model_name}...')
pd.DataFrame(list(homogenized_embeddings.items()), columns = ['Entity', 'Embedding']).to_pickle(f'stupid.pkl')