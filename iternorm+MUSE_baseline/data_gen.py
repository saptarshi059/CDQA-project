#python data_gen.py -b phiyodr/bert-base-finetuned-squad2 

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, logging
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bert_variant', required=True)
args = parser.parse_args()

logging.set_verbosity(50)

BERT_variant = args.bert_variant
tokenizer = AutoTokenizer.from_pretrained(BERT_variant)
model = AutoModel.from_pretrained(BERT_variant)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'{args.bert_variant} loaded on device: {device}')

model_embeddings = model.get_input_embeddings()

def aggregate_vals(df, data_type):
	embs = []
	if data_type == 'test':
		for row in tqdm(df.itertuples(index=False)):
			embs.append(np.array(list(row[2:])))
	elif data_type == 'train':
		for row in tqdm(df.itertuples(index=False)):
			embs.append(np.array(list(row[1:])))

	df['source_embedding'] = embs
	df.drop(columns=range(1,51), inplace=True)

	return df

umls_embedding_folder = os.path.abspath('../umls_embedding')

embeddings = pd.read_csv(os.path.join(umls_embedding_folder,'embeddings.csv'),header=None)
our_cui_pc = pd.read_csv(os.path.join(umls_embedding_folder,'CUI_PC.csv'))

print('Files loaded...')

###Test Data Gen###

#Renaming for merging
embeddings.rename(columns={0:'CUI'}, inplace=True)
our_cui_pc.rename(columns={'Preferred_Concept':'PC'}, inplace=True) #Aesthetics!

test_data = aggregate_vals(our_cui_pc.merge(embeddings, on='CUI', how='inner'), data_type='test')
test_data.drop(columns=['CUI'], inplace=True) #Cause we don't need the CUI's anymore for this method

avg_sw_embs = []

for row in tqdm(test_data.itertuples(index=False)):
    entity_tokens = tokenizer(row.PC, return_tensors='pt', max_length=512, truncation=True, padding='max_length')['input_ids'][0]
    sw_embds = []
    for index in range(1, len(entity_tokens)-1):
        sw_embds.append(model_embeddings(entity_tokens[index].to(device)))
    avg_sw_embs.append(torch.mean(torch.vstack(sw_embds), dim=0).cpu().detach().numpy())

test_data['target_embedding'] = avg_sw_embs
test_data.to_pickle('pre_norm_data.pkl')
print('Embedding table for projection generated...')