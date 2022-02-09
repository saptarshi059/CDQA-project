#python def_embed.py -b phiyodr/bert-base-finetuned-squad2 -av 0

from transformers import AutoModel, AutoTokenizer, logging
import argparse
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm
import torch

logging.set_verbosity(50)

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bert_variant', required=True)
parser.add_argument('-av', '--avg_emb', required=True, type=int)
args = parser.parse_args()

model = AutoModel.from_pretrained(args.bert_variant)
tokenizer = AutoTokenizer.from_pretrained(args.bert_variant)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'{args.bert_variant} loaded on device: {device}')

dataframe = pickle.load(open('Entity_Definition.pkl', 'rb'))
definition_embeddings = []
avg_emb = []
for row in tqdm(dataframe.itertuples(index=False)):
	inputs = tokenizer(row.Definition, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
	inputs.to(device)
	output = model(**inputs)
	if args.avg_emb == 1:
		definition_embeddings.append(output['pooler_output'].detach().cpu())
	else:
		avg_emb.append(torch.mean(torch.vstack([row.UMLS_Embedding, output['pooler_output'].detach().cpu()]), dim=0).reshape(1,-1))

if args.avg_emb == 1:
	#pd.DataFrame(zip(dataframe.Entity, dataframe.UMLS_Embedding, definition_embeddings), columns=['Entity', 'UMLS_Embedding', 'Dictionary_Embedding']).to_pickle(f"NN-DTE-to-{args.bert_variant.replace('/','-')}.pkl")
	pd.DataFrame(zip(dataframe.Entity, definition_embeddings), columns=['Entity', 'UMLS_Embedding']).to_pickle(f"NN-DTE-to-{args.bert_variant.replace('/','-')}.pkl") #Trying only definition embeddings
else:
	#I'm just calling it UMLS_Embedding so that it runs easily with run_modelling.py
	pd.DataFrame(zip(dataframe.Entity, avg_emb), columns=['Entity', 'UMLS_Embedding']).to_pickle(f"NN-DTE-to-{args.bert_variant.replace('/','-')}.pkl")