#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reading in the dataset
import json
import pandas as pd

def read_covidqa():
    with open('COVID-QA.json', 'rb') as f:
        covidqa_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in covidqa_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    
    return contexts, questions, answers

all_contexts, all_questions, all_answers = read_covidqa()

#Converting to a dataframe for easy k-fold splits
full_dataset = pd.DataFrame(list(zip(all_contexts, all_questions, all_answers)), columns =['context', 'question', 'answer'])


# In[2]:


def preprocess_input(dataset):
    answers = dataset['answer'].to_list()
    context = dataset['context'].to_list()
    
    for answer, context in zip(dataset['answer'], dataset['context']):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters
    
    encodings = tokenizer(dataset['context'].to_list(), dataset['question'].to_list(),                           truncation=True, padding=True)
    
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        
        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    
    return encodings


# In[3]:


#Code to compute F1 & EM scores
import re
import string
import collections
import numpy as np

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_EM(df):
    EM = []
    for i in range(len(df)):
        a_gold = df['true_answer'][i]
        a_pred = df['predicted_answer'][i]
        EM.append(int(normalize_answer(a_gold) == normalize_answer(a_pred)))
    return np.mean(EM)

def compute_f1_main(df):
    F1 = []
    for i in range(len(df)):
        a_gold = df['true_answer'][i]
        a_pred = df['predicted_answer'][i]
        F1.append(compute_f1(a_gold,a_pred))
    return np.mean(F1)


# In[4]:


#Dataloader Object Creation
import torch

class CovidQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


# In[5]:


#Main fine-tuning loop
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
from tqdm import tqdm

kfold = KFold(n_splits=5)
num_epochs = 3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_name = 'navteca/roberta-base-squad2'

tokenizer = AutoTokenizer.from_pretrained(model_name)

fold_F1_score = []
fold_EM_score = []

for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)): 
    print(f'FOLD {fold}')
    print('--------------------------------')
     
    train_dataset = CovidQADataset(preprocess_input(full_dataset.iloc[train_ids]))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    
    model.zero_grad()
    model.init_weights()
    
    model.train()
    
    # Initialize optimizer
    optim = AdamW(model.parameters(), lr=5e-5)
    
    # Run the training loop for defined number of epochs
    for epoch in tqdm(range(3)):
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,                             end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optim.step()

    # Print about testing
    print('Starting testing')

    # Evaluationfor this fold
    with torch.no_grad():
        model.eval()
        test_data = full_dataset.iloc[test_ids]
        nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=-1 if device == torch.device('cpu')                                else 0)
        questions = []
        true_answers = []
        predicted_answers = []
        final_df = pd.DataFrame(columns=['question', 'true_answer', 'predicted_answer'])

        # Iterate over the test data and generate predictions
        for i in range(len(test_data)):
            context = test_data.iloc[i]['context']
            questions.append(test_data.iloc[i]['question'])
            true_answers.append(test_data.iloc[i]['answer']['text'])

            # Generate outputs
            QA_input = {'question': questions[i], 'context': context}
            predicted_answers.append(nlp(QA_input)['answer'])
        
        final_df['question'] = questions
        final_df['true_answer'] = true_answers
        final_df['predicted_answer'] = predicted_answers
            
    #Print F1
    fold_F1_score.append(compute_f1_main(final_df))
    print(f'F1 for fold {fold}: {fold_F1_score[fold]}')
    
    #Print EM
    fold_EM_score.append(compute_EM(final_df))
    print(f'EM for fold {fold}: {fold_EM_score[fold]}')

print(f"Avg. F1: {np.mean(fold_F1_score)}")
print(f"Avg. EM: {np.mean(fold_EM_score)}")

