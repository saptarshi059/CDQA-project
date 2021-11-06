import pandas as pd
import numpy as np
from tqdm import tqdm

def aggregate_vals(df, data_type):
	embs = []
	if data_type == 'test':
		for row in tqdm(df.itertuples(index=False)):
			embs.append(np.array(list(row[2:])))
	elif data_type == 'train':
		for row in tqdm(df.itertuples(index=False)):
			embs.append(np.array(list(row[1:])))

	df['Embedding'] = embs
	df.drop(columns=range(1,51), inplace=True)

	return df

embeddings = pd.read_csv('embeddings.csv',header=None)
our_cui_pc = pd.read_csv('CUI_PC.csv')
their_CUI_PC = pd.read_csv('their_CUI_PC.csv')

print('Files loaded...')

###Test Data Gen###

#Renaming for merging
embeddings.rename(columns={0:'CUI'}, inplace=True)
our_cui_pc.rename(columns={'Preferred_Concept':'PC'}, inplace=True) #Aesthetics!

test_data = aggregate_vals(our_cui_pc.merge(embeddings, on='CUI', how='inner'), data_type='test')

###Train Data Gen###
#Removing the common CUI's from the embeddings dataframe
embeddings.set_index('CUI', inplace=True)
embeddings.drop(test_data.CUI.to_list(), axis=0, inplace=True)
embeddings.reset_index(inplace=True)

embeddings = aggregate_vals(embeddings, data_type='train')
train_data = embeddings.merge(their_CUI_PC, on='CUI', how='inner') #Collecting the PC's for the CUI's

test_data.drop(columns='CUI', inplace=True)
test_data.to_pickle('test_data.pkl')
print('Test data saved...')

train_data.drop(columns='CUI', inplace=True)
train_data.to_pickle('training_data.pkl')
print('Training data saved...')