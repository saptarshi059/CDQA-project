__author__ = 'Connor Heaton and Saptarshi Sengupta'

import re
import json
import time
import torch
import string
import argparse
import collections

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from transformers import AdamW, AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline

# from custom_question_rep import custom_question_rep_gen


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


def read_covidqa(fp):
    with open(fp, 'rb') as f:
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


def preprocess_input(dataset, tokenizer_):
    answers = dataset['answer'].to_list()
    context = dataset['context'].to_list()

    question_texts = dataset['question'].to_list()
    context_texts = dataset['context'].to_list()

    for answer, context in zip(dataset['answer'], dataset['context']):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters

    encodings = tokenizer_(dataset['context'].to_list(), dataset['question'].to_list(), \
                           truncation=True, padding=True)

    # for k, v in encodings.items():
    #     print('k: {} v: {}'.format(k, len(v)))

    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer_.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer_.model_max_length

    # print('start_positions: {}'.format(start_positions))
    # print('end_positions: {}'.format(end_positions))

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions,
                      'question_texts': question_texts, 'context_texts': context_texts})
    # for k, v in encodings.items():
    #     print('k: {} v: {}'.format(k, len(v)))
    # input('encodings: {}'.format(encodings.keys()))

    return encodings


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
    if not s:
        return []
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
        F1.append(compute_f1(a_gold, a_pred))
    return np.mean(F1)


class CovidQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) if key not in ['question_texts', 'context_texts'] else val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='COVID-QA.json', help='Filepath to CovidQA dataset')

    parser.add_argument('--n_splits', default=5, help='How many folds to use for cross val', type=int)
    parser.add_argument('--batch_size', default=32, help='How many items to process as once', type=int)
    parser.add_argument('--lr', default=5e-5, help='How many items to process as once', type=float)
    parser.add_argument('--n_epochs', default=3, help='If training/fine-tuning, how many epochs to perform')
    parser.add_argument('--model_name', default='navteca/roberta-base-squad2',
                        help='Type of model to use from HuggingFace')

    parser.add_argument('--use_kge', default=False, help='If KGEs should be place in input')

    args = parser.parse_args()

    USE_KGE = args.use_kge
    kfold = KFold(n_splits=args.n_splits)
    all_contexts, all_questions, all_answers = read_covidqa(args.data)
    # Converting to a dataframe for easy k-fold splits
    full_dataset = pd.DataFrame(list(zip(all_contexts, all_questions, all_answers)),
                                columns=['context', 'question', 'answer'])
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    fold_f1_score = []
    fold_EM_score = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_dataset = CovidQADataset(preprocess_input(full_dataset.iloc[train_ids], tokenizer))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # input('okty')

        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        model.to(device)
        model.train()
        optim = AdamW(model.parameters(), lr=args.lr)

        # Run the training loop for defined number of epochs
        for epoch_idx in range(args.n_epochs):
            for batch_idx, batch in enumerate(train_loader):
                batch_start_time = time.time()
                optim.zero_grad()
                question_texts = batch['question_texts']
                context_texts = batch['context_texts']
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                if USE_KGE:
                    input_embds, offsets = [], []
                    for q_text, c_text in zip(question_texts, context_texts):
                        this_input_embds, this_n_token_adj = custom_question_rep_gen(q_text, c_text)

                        input_embds.append(this_input_embds)
                        offsets.append(this_n_token_adj)

                    input_embds = torch.cat(input_embds, dim=0)
                    offsets = torch.cat(offsets, dim=0)

                    start_positions = start_positions - offsets
                    end_positions = end_positions - offsets

                else:
                    model_embds = model.get_input_embeddings()
                    input_embds = model_embds(input_ids)

                outputs = model(inputs_embeds=input_embds, attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optim.step()
                batch_elapsed_time = time.time() - batch_start_time
                print('Epoch: {0}/{1} Iter: {2} Loss: {3:.4f} Time: {4:.2f}s'.format(epoch_idx,
                                                                                     args.n_epochs,
                                                                                     batch_idx,
                                                                                     loss, batch_elapsed_time))

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Evaluationfor this fold
        test_data = full_dataset.iloc[test_ids]
        nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=-1 if device == torch.device('cpu') \
            else 0)
        with torch.no_grad():
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

        # Print F1
        fold_f1_score.append(compute_f1_main(final_df))
        print(f'F1 for fold {fold}: {fold_f1_score[fold]}')

        # Print EM
        fold_EM_score.append(compute_EM(final_df))
        print(f'EM for fold {fold}: {fold_EM_score[fold]}')

    print(f"Avg. F1: {np.mean(fold_f1_score)}")
    print(f"Avg. EM: {np.mean(fold_EM_score)}")
