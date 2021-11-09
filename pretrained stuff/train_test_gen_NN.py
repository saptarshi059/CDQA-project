#python train_test_gen_NN.py -b phiyodr/bert-base-finetuned-squad2 -d training_data.pkl
#python train_test_gen_NN.py -b phiyodr/bert-base-finetuned-squad2 -d test_data.pkl
#navteca/roberta-base-squad2

import argparse
import pickle5 as pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel, logging
import torch
from tqdm import tqdm
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--bert_variant', required=True)
parser.add_argument('-d', '--data_file', required=True)
parser.add_argument('-nos', '--number_of_samples', default=10000, type=int)
parser.add_argument('-Th', '--through', type=str2bool, default=False)

args = parser.parse_args()

logging.set_verbosity(50)

BERT_variant = args.bert_variant
tokenizer = AutoTokenizer.from_pretrained(BERT_variant)
model = AutoModel.from_pretrained(BERT_variant)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Model loaded on device: {device}')

model_embeddings = model.get_input_embeddings()

df = pickle.load(open(args.data_file, 'rb'))
df = df.iloc[0:args.number_of_samples] #Taking the first nos samples

print('Loaded all necessary files & processed DataFrame...')

src = []
tgt = []
PC = []
CUI = []
for row in tqdm(df.itertuples(index=False)):
    CUI.append(row.CUI)
    PC.append(row.PC)
    src.append(row.Embedding)
    if args.through == False:
        entity_tokens = tokenizer(row.PC, return_tensors='pt')['input_ids'][0]
        sw_embds = []
        for index in range(1, len(entity_tokens)-1):
            sw_embds.append(model_embeddings(entity_tokens[index].to(device)))
        tgt.append(torch.mean(torch.vstack(sw_embds), dim=0).cpu().detach().numpy())
    else:    
        inputs = tokenizer(row.PC, return_tensors='pt')
        inputs.to(device)
        try:
            output = model(**inputs)
            tgt.append(output['pooler_output'].detach().cpu().numpy()[0])
        except:
            del PC[-1]
            del src[-1]

if 'train' in args.data_file:
    pd.DataFrame(zip(src, tgt), columns=['train', 'test']).to_pickle(f'{args.data_file.replace(".pkl","")}_Homogenization_data.pkl')
else:
    pd.DataFrame(zip(CUI, PC, src, tgt), columns=['CUI', 'PC', 'train', 'test']).to_pickle(f'{args.data_file.replace(".pkl","")}_Homogenization_data.pkl')

print('Dataset created...')