#python def_embed.py -b phiyodr/bert-base-finetuned-squad2

from transformers import AutoModel, AutoTokenizer, logging
import argparse
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm
import torch

logging.set_verbosity(50)

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bert_variant', required=True)
args = parser.parse_args()

model = AutoModel.from_pretrained(args.bert_variant)
tokenizer = AutoTokenizer.from_pretrained(args.bert_variant)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'{args.bert_variant} loaded on device: {device}')

dataframe = pickle.load(open('Entity_Definition.pkl', 'rb'))
definition_embeddings = []
for row in tqdm(dataframe.itertuples(index=False)):
	inputs = tokenizer(row.Definition, return_tensors='pt', truncation=True)
	inputs.to(device)
	output = model(**inputs)
	definition_embeddings.append(output['pooler_output'].detach().cpu().numpy()[0])

pd.DataFrame(zip(dataframe.Entity, dataframe.UMLS_Embedding, definition_embeddings), columns=['Entity', 'UMLS_Embedding', 'Dictionary_Embedding']).to_pickle(f"NN-DTE-to-{args.bert_variant.replace('/','-')}.pkl")