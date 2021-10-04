#!/usr/bin/env python
# coding: utf-8

import json
from bert_score import score
from transformers import AutoModelForQuestionAnswering, QuestionAnsweringPipeline, AutoTokenizer
import torch
from tqdm import tqdm

def read_covidqa():
    with open('COVID-QA_cleaned.json', 'rb') as f:
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
                    answers.append(answer['text'])
    
    return contexts, questions, answers

all_contexts, all_questions, all_answers = read_covidqa()
print('Dataset Loaded...')

def gen_answers(model_name):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f'Model & Tokenizer loaded for: {model_name}...')
    
    if torch.cuda.is_available():
        print('Using Device: GPU')
    else:
        print('Using Device: CPU')

    nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=torch.cuda.current_device())
    
    predicted_answers = []
    
    with torch.no_grad():
        for q,c in tqdm(zip(all_questions, all_contexts)):
            QA_input = {'question': q, 'context': c}
            predicted_answers.append(nlp(QA_input)['answer'])

    return predicted_answers

'''
bert_answers = gen_answers('phiyodr/bert-base-finetuned-squad2')
P, R, F1 = score(bert_answers, all_answers, model_type='bert-base-uncased' ,verbose=True)
print(f'F1 score of phiyodr/bert-base-finetuned-squad2 with bert_score: {F1.mean():.3f}')

scibert_answers = gen_answers('ktrapeznikov/scibert_scivocab_uncased_squad_v2')
P, R, F1 = score(scibert_answers, all_answers, lang='en-sci', verbose=True)
print(f'F1 score of ktrapeznikov/scibert_scivocab_uncased_squad_v2 with bert_score: {F1.mean():.3f}')
'''

def incorrect_compute(model_name, model_answers):
    model_incorrect = {}
    for index, (Q, TA, PA) in tqdm(enumerate(zip(all_questions, all_answers, model_answers))):
        if TA != PA:
            model_incorrect[index] = (Q, TA, PA)

    print(f'{model_name} got {len(model_incorrect)} questions wrong...')

    print('Saving incorrect answers...')
    mn = model_name.replace('/','_')
    with open(f'{mn}_incorrect.txt', 'w') as f:
        f.write(json.dumps(model_incorrect))

#incorrect_compute('phiyodr/bert-base-finetuned-squad2', bert_answers)
#incorrect_compute('ktrapeznikov/scibert_scivocab_uncased_squad_v2', scibert_answers)

bert_answers = gen_answers('phiyodr/bert-base-finetuned-squad2')
scibert_answers = gen_answers('ktrapeznikov/scibert_scivocab_uncased_squad_v2')

def save_answers(model_name, predicted_answers):
    full_set = {}
    for Q, TA, PA in tqdm(zip(all_questions, all_answers, predicted_answers)):
        full_set[Q] = (TA, PA)

    mn = model_name.replace('/','_')
    with open(f'{mn}_answers.txt', 'w') as f:
        f.write(json.dumps(full_set))

save_answers('phiyodr/bert-base-finetuned-squad2', bert_answers)
save_answers('ktrapeznikov/scibert_scivocab_uncased_squad_v2', scibert_answers)