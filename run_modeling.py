__author__ = 'Connor Heaton and Saptarshi Sengupta'

import re
import gc
import os
import json
import math
import time
import torch
import string
import random
import argparse
import datetime
import collections

import numpy as np
import pandas as pd
import pickle5 as pickle

import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from bert_score import score
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from transformers import get_constant_schedule_with_warmup
from transformers import AdamW, AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
# from custom_question_rep import custom_question_rep_gen

from tqdm import tqdm
from input_maker import InputMaker
# from custom_input import custom_input_rep
# from custom_qa_pipeline import CustomQuestionAnsweringPipeline
from distributed_fold_trainer import DistributedFoldTrainer

from datasets import CovidQADataset


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
    # with open(fp, 'rb') as f:
    #     covidqa_dict = json.load(f)

    with open(fp, encoding='utf-8', errors='ignore') as json_data:
        covidqa_dict = json.load(json_data, strict=False)

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


def preprocess_input(dataset, tokenizer_, n_stride=64, max_len=512, n_neg=1, maker=None):
    answers = dataset['answer'].to_list()
    context = dataset['context'].to_list()

    question_texts = dataset['question'].to_list()
    context_texts = dataset['context'].to_list()
    # print('len(question_texts): {}'.format(len(question_texts)))
    # print('len(context_texts): {}'.format(len(context_texts)))

    pad_on_right = tokenizer_.padding_side == "right"

    # ensure start and end indices are accurate
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

    # if maker is not None:
    #     print('Converting questions using custom domain term questions...')
    #     all_questions_ = dataset['question'].tolist()
    #     all_questions_ = [maker.convert_questions_to_kge(q) for q in all_questions_]
    #     dataset['question'] = all_questions_

    print('*** preprocess_input ***')
    print('n_stride: {}'.format(n_stride))
    print('max_len: {}'.format(max_len))
    print('************************')

    q_lens = [len(q) for q in dataset['question'].to_list()]
    max_q_len = max(q_lens)
    min_q_len = min(q_lens)
    print('min_q_len: {} max_q_len: {}'.format(min_q_len, max_q_len))

    c_lens = [len(c) for c in dataset['context'].to_list()]
    max_c_len = max(c_lens)
    min_c_len = min(c_lens)
    print('min_c_len: {} max_c_len: {}'.format(min_c_len, max_c_len))

    encodings = tokenizer_(
        dataset['question'].to_list() if pad_on_right else dataset['context'].to_list(),
        dataset['context'].to_list() if pad_on_right else dataset['question'].to_list(),
        truncation='longest_first',
        stride=n_stride,
        padding=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        max_length=max_len,
    )
    # print('encodings.keys(): {}'.format(encodings.keys()))
    # print('encodings[input_ids]: {}'.format(len(encodings['input_ids'])))

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = encodings.pop("overflow_to_sample_mapping")
    # print('sample_mapping: {}'.format(len(sample_mapping)))
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = encodings.pop("offset_mapping")
    # print('offset_mapping: {}'.format(len(offset_mapping)))

    # input('sample_mapping: {}'.format(sample_mapping))
    # input('offset_mapping: {}'.format(offset_mapping))

    encodings["start_positions"] = []
    encodings["end_positions"] = []

    positive_idxs = []
    neg_idxs_by_sample = {}

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = encodings["input_ids"][i]
        cls_index = input_ids.index(tokenizer_.cls_token_id)
        # print('offsets: {}'.format(offsets))
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = encodings.sequence_ids(i)
        # print('sequence_ids: {}'.format(sequence_ids))

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        # print('dataset[\'answer\']: {}'.format(dataset['answer']))
        # answers = dataset['answer'][sample_index]
        i_answers = answers[sample_index]
        # print('i_answers: {}'.format(i_answers))

        # If no answers are given, set the cls_index as answer.
        if i_answers["answer_start"] is None:
            # input('i_answers["answer_start"]: {}'.format(i_answers["answer_start"]))
            encodings["start_positions"].append(cls_index)
            encodings["end_positions"].append(cls_index)

            curr_neg_idxs_for_sample = neg_idxs_by_sample.get(sample_index, [])
            curr_neg_idxs_for_sample.append(i)
            neg_idxs_by_sample[sample_index] = curr_neg_idxs_for_sample
        else:

            # Start/end character index of the answer in the text.
            start_char = i_answers["answer_start"]
            end_char = start_char + len(i_answers["text"])
            # print('start_char: {}'.format(start_char))
            # print('end_char: {}'.format(end_char))

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            # print('** token_start_index: {} **'.format(token_start_index))

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            # print('** token_end_index: {} **'.format(token_end_index))

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                # print('offsets[token_start_index]: {}'.format(offsets[token_start_index]))
                # print('offsets[token_end_index]: {}'.format(offsets[token_end_index]))
                encodings["start_positions"].append(cls_index)
                encodings["end_positions"].append(cls_index)

                curr_neg_idxs_for_sample = neg_idxs_by_sample.get(sample_index, [])
                curr_neg_idxs_for_sample.append(i)
                neg_idxs_by_sample[sample_index] = curr_neg_idxs_for_sample
                # input('appending cls index')
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                encodings["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                encodings["end_positions"].append(token_end_index + 1)

                positive_idxs.append(i)

    # print('encodings.keys: {}'.format(encodings.keys()))
    # for k, v in encodings.items():
    #     if type(v) == list:
    #         print('k: {} v: {}'.format(k, len(v)))

    encodings["question_texts"] = []
    encodings["context_texts"] = []
    for i in range(len(encodings['input_ids'])):

        i_question_text = question_texts[sample_mapping[i]]
        i_input_ids = encodings['input_ids'][i]
        i_input_text = tokenizer_.decode(i_input_ids, skip_special_tokens=True)
        if i_input_text.startswith(i_question_text):
            i_context_text = i_input_text[len(i_question_text):]
        else:
            i_context_text = i_input_text[:-len(i_question_text)]

        encodings["question_texts"].append(i_question_text)
        encodings["context_texts"].append(i_context_text)

    for k, v in encodings.items():
        print('k: {} v: {}'.format(k, len(v)))

    print('len(positive_idxs): {}'.format(len(positive_idxs)))
    print('Selecting up to {} negative records for each sample...'.format(n_neg))

    for sample_idx, potential_neg_idxs in neg_idxs_by_sample.items():
        if n_neg > 0:
            selected_neg_idxs = random.choices(potential_neg_idxs, k=n_neg)
        else:
            selected_neg_idxs = potential_neg_idxs
        positive_idxs.extend(selected_neg_idxs)

    positive_idxs = list(sorted(positive_idxs))
    print('len(positive_idxs): {}'.format(len(positive_idxs)))

    for encoding_key in encodings.keys():
        encodings[encoding_key] = [encodings[encoding_key][i] for i in positive_idxs]

    for k, v in encodings.items():
        print('k: {} v: {}'.format(k, len(v)))

    # input('okty')

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





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        default='data/COVID-QA_cleaned_final.json',
                        # default='data/200423_covidQA.json',
                        help='Filepath to CovidQA dataset')
    parser.add_argument('--out', default='out', help='Directory to put output')

    parser.add_argument('--n_splits', default=5, help='How many folds to use for cross val', type=int)
    parser.add_argument('--batch_size', default=40, help='How many items to process as once', type=int)
    parser.add_argument('--lr', default=5e-5, help='How many items to process as once', type=float)
    parser.add_argument('--n_epochs', default=3, help='If training/fine-tuning, how many epochs to perform', type=int)
    parser.add_argument('--n_stride', default=196, type=int)
    parser.add_argument('--max_len', default=384, type=int)
    parser.add_argument('--model_name',
                        # default='ktrapeznikov/scibert_scivocab_uncased_squad_v2',
                        # default='clagator/biobert_squad2_cased',
                        default='navteca/roberta-base-squad2',
                        # default='phiyodr/bert-base-finetuned-squad2',
                        # default='ktrapeznikov/biobert_v1.1_pubmed_squad_v2',
                        # default='ktrapeznikov/scibert_scivocab_uncased_squad_v2',
                        help='Type of model to use from HuggingFace')

    parser.add_argument('--use_kge', default=False, help='If KGEs should be place in input',
                        type=str2bool)
    parser.add_argument('--use_dict', default=False, help='If KGEs should be place in input',
                        type=str2bool)
    parser.add_argument('--concat_kge', default=False, type=str2bool)
    parser.add_argument('--fancy_concat', default=False, type=str2bool)
    parser.add_argument('--random_kge', default=False, type=str2bool)
    parser.add_argument('--seed', default=16, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, help='Fuck Timo Moller', type=float)
    parser.add_argument('--vanilla_adam', default=False, type=str2bool)

    parser.add_argument('--dte_lookup_table_fp',
                        default='NN-DTE-to-phiyodr-bert-base-finetuned-squad2.pkl'
                        # default='DTE_to_phiyodr_bert-base-finetuned-squad2.pkl',
                        # default='DTE_to_ktrapeznikov_biobert_v1.1_pubmed_squad_v2.pkl',
                        # default='DTE_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl'
                        )
    parser.add_argument('--n_neg_records', default=1, type=int)

    parser.add_argument('--gpus', default=[0], help='Which GPUs to use', type=int, nargs='+')
    parser.add_argument('--port', default='14345', help='Port to use for DDP')

    args = parser.parse_args()
    mp.set_sharing_strategy('file_system')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    torch.manual_seed(args.seed)
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print('*' * len('* Model time ID: {} *'.format(curr_time)))
    print('* Model time ID: {} *'.format(curr_time))
    print('*' * len('* Model time ID: {} *'.format(curr_time)))

    my_maker = InputMaker(args)

    model_outdir = os.path.join(args.out, curr_time)
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)

    model_ckpt_dir = os.path.join(model_outdir, 'models')
    if not os.path.exists(model_ckpt_dir):
        os.makedirs(model_ckpt_dir)

    tb_dir = os.path.join(model_outdir, 'tb_dir')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    model_ckpt_tmplt = os.path.join(model_ckpt_dir, 'model_fold{}.pt')

    model_arg_fp = os.path.join(model_outdir, 'args.txt')
    args_d = vars(args)
    with open(model_arg_fp, 'w+') as f:
        for k, v in args_d.items():
            f.write('{} = {}\n'.format(k, v))

    USE_KGE = args.use_kge or args.use_dict
    effective_model_name = args.model_name.replace('/', '-')
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_out_fname = '{}{}_{}.txt'.format(effective_model_name,
                                           '_kge' if USE_KGE else '',
                                           curr_time)
    model_out_fp = os.path.join(model_outdir, model_out_fname)
    print('*** model_out_fp: {} ***'.format(model_out_fp))

    DTE_Model_Lookup_Table = pickle.load(open(args.dte_lookup_table_fp, 'rb'))
    dtes = []
    umls_dtes = DTE_Model_Lookup_Table['UMLS_Embedding'].tolist()
    dict_dtes = DTE_Model_Lookup_Table['Dictionary_Embedding'].tolist()
    
    if args.use_kge:
        dtes.extend(umls_dtes)
    if args.use_dict:
        dtes.extend(dict_dtes)
    
    if args.random_kge:
        print('Replacing DTEs with random tensors...')
        dtes = [torch.rand(1, 768) for _ in dtes]
    print('dtes[0]: {}'.format(dtes[0]))
    dtes = torch.cat(dtes, dim=0).to('cpu')
    
    custom_domain_term_tokens = []
    domain_terms = DTE_Model_Lookup_Table['Entity'].tolist()
    custom_umls_tokens = ['[{}]'.format(dt) for dt in domain_terms]
    custom_dict_tokens = ['#{}#'.format(dt) for dt in domain_terms]
    if args.use_kge:
        custom_domain_term_tokens.extend(custom_umls_tokens)
    if args.use_dict:
        custom_domain_term_tokens.extend(custom_dict_tokens)
    # input('dtes: {}'.format(dtes.shape))

    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    all_contexts, all_questions, all_answers = read_covidqa(args.data)
    if USE_KGE:
        all_questions = [my_maker.convert_questions_to_kge(q) for q in all_questions]
    # input('okty')
    # Converting to a dataframe for easy k-fold splits
    full_dataset = pd.DataFrame(list(zip(all_contexts, all_questions, all_answers)),
                                columns=['context', 'question', 'answer'])

    # sample_encoded_inputs = tokenizer(text='the first string of text',
    #                                   text_pair='the second string of text')
    # input('sample_encoded_inputs: {}'.format(sample_encoded_inputs))

    N_STRIDE = args.n_stride
    # MAX_LEN = tokenizer.model_max_length if tokenizer.model_max_length <= 512 else 512
    MAX_LEN = args.max_len
    # some of the tokenizers return 1000000000000000019884624838656 as model_max_length for some reason

    fold_f1_score = []
    fold_f1_score_with_bert_score = []
    fold_EM_score = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
        print('FOLD {}'.format(fold))
        print('--------------------------------')
        model_ckpt_fp = model_ckpt_tmplt.format(fold)
        dtes = dtes.to('cpu')
        print('\t$$$ dtes[:10, :10]: {} $$$'.format(dtes[:10, :10]))

        print('Preparing dataset for fold...')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if USE_KGE:
            print('Adding {} custom domain tokens to tokenizer...'.format(len(custom_domain_term_tokens)))
            print('\tcustom_domain_term_tokens[:6]: {}'.format(custom_domain_term_tokens[:6]))
            print('\tcustom_domain_term_tokens[-6:]: {}'.format(custom_domain_term_tokens[-6:]))
            tokenizer.add_tokens(custom_domain_term_tokens)

        dataset = CovidQADataset(preprocess_input(full_dataset.iloc[train_ids], tokenizer,
                                                  n_stride=N_STRIDE, max_len=MAX_LEN, n_neg=args.n_neg_records))

        del tokenizer
        dist_arg_d = {
            'model_ckpt_fp': model_ckpt_fp,
            'tb_dir': tb_dir,
            'dataset': dataset,
            'train_ids': train_ids,
            'model_name': args.model_name,
            'N_STRIDE': N_STRIDE,
            'MAX_LEN': MAX_LEN,
            'world_size': len(args.gpus),
            'batch_size': args.batch_size,
            'lr': args.lr,
            'n_epochs': args.n_epochs,
            'USE_KGE': USE_KGE,
            'fold': fold,
            'n_splits': args.n_splits,
            'n_neg_records': args.n_neg_records,
            'dtes': dtes,
            'warmup_proportion': args.warmup_proportion,
            'seed': args.seed,
            'concat_kge': args.concat_kge,
            'my_maker': my_maker,
            'args': args,
        }

        print('Training {} distributed model(s) for fold {}...'.format(len(args.gpus), fold))
        # mp.spawn(train_fold_distributed, nprocs=len(args.gpus), args=(model_ckpt_fp, tb_dir, full_dataset, train_ids,
        #                                                               args.model_name, N_STRIDE, MAX_LEN,
        #                                                               len(args.gpus), args.batch_size,
        #                                                               args.lr, args.n_epochs, USE_KGE,
        #                                                               fold, args.n_splits, args.n_neg_records,
        #                                                               dtes,
        #                                                               args.warmup_proportion,
        #                                                               args.seed))
        mp.spawn(DistributedFoldTrainer, nprocs=len(args.gpus), args=(dist_arg_d, ))
        # input('okty')
        # Process is complete.
        print('Training process has finished...')

        print('Loading trained model ckpt...')
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        model = model.to(device)
        dtes = dtes.to(model.device)

        if USE_KGE:
            initial_input_embeddings = model.get_input_embeddings().weight
            new_input_embedding_weights = torch.cat([initial_input_embeddings, dtes], dim=0)
            new_input_embeddings = nn.Embedding.from_pretrained(new_input_embedding_weights, freeze=False)
            model.set_input_embeddings(new_input_embeddings)

        map_location = {'cuda:0': 'cuda:0'}
        state_dict = torch.load(model_ckpt_fp, map_location=map_location)
        model.load_state_dict(state_dict)
        model.eval()

        # Print about testing
        print('Starting testing')

        # Evaluationfor this fold
        test_data = full_dataset.iloc[test_ids]
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if USE_KGE:
            print('Adding {} custom domain tokens to tokenizer...'.format(len(custom_domain_term_tokens)))
            print('\tcustom_domain_term_tokens[:6]: {}'.format(custom_domain_term_tokens[:6]))
            print('\tcustom_domain_term_tokens[-6:]: {}'.format(custom_domain_term_tokens[-6:]))
            tokenizer.add_tokens(custom_domain_term_tokens)

        nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer,
                                        device=-1 if device == torch.device('cpu') else 0)
        with torch.no_grad():
            questions = []
            true_answers = []
            predicted_answers = []
            final_df = pd.DataFrame(columns=['question', 'true_answer', 'predicted_answer'])

            # Iterate over the test data and generate predictions
            for i in range(len(test_data)):
                # if i > 2:
                #     break
                context = test_data.iloc[i]['context']
                questions.append(test_data.iloc[i]['question'])
                true_answers.append(test_data.iloc[i]['answer']['text'])

                # Generate outputs
                q_i = questions[i]
                # if USE_KGE:
                #     q_i = my_maker.convert_questions_to_kge(q_i)
                QA_input = {'question': q_i, 'context': context}
                input_embds = None
                attn_mask = None
                input_ids = None
                predicted_answer = nlp(
                    QA_input,
                    # _input_ids_=input_ids,
                    # _input_embds_=None,
                    # _attention_mask_=attn_mask,
                    max_seq_len=MAX_LEN,
                    doc_stride=N_STRIDE
                )['answer']

                predicted_answers.append(predicted_answer)

            final_df['question'] = questions
            final_df['true_answer'] = true_answers
            final_df['predicted_answer'] = predicted_answers

        # Print F1
        fold_f1_score.append(compute_f1_main(final_df))
        print('F1 for fold {}: {}'.format(fold, fold_f1_score[fold]))

        P, R, F1 = score(predicted_answers, true_answers, lang='en')
        fold_f1_score_with_bert_score.append(F1.mean().item())
        print(f'F1 score for fold {fold} with bert_score: {F1.mean().item()}')

        # Print EM
        fold_EM_score.append(compute_EM(final_df))
        print('EM for fold {}: {}'.format(fold, fold_EM_score[fold]))

        del model
        del nlp
        del final_df
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("Avg. F1: {}".format(np.mean(fold_f1_score)))
    print("Avg. F1 with BERTScore: {}".format(np.mean(fold_f1_score_with_bert_score)))
    print("Avg. EM: {}".format(np.mean(fold_EM_score)))

    print('Writing results to file...')
    write_lines = ['Fold: {0}\tF1: {1:.4f}\tEM: {2:.4}'.format(f_idx, f1, em)
                   for f_idx, (f1, em) in enumerate(zip(fold_f1_score, fold_EM_score))]
    write_lines.append('Avg. F1: {}'.format(np.mean(fold_f1_score)))
    write_lines.append('Avg. EM: {}'.format(np.mean(fold_EM_score)))

    with open(model_out_fp, 'w+') as f:
        f.write('\n'.join(write_lines))
