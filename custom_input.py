# Extracting embeddings for non-domain terms. I'm simply using BERT's tokenizer for the nDT's.
# Creating question representations in this block.
from transformers import AutoTokenizer, AutoModel
import torch
import re
import os
import pandas as pd
import re

import pickle5 as pickle

# DTE_Model_Lookup_Table = pd.read_pickle(os.path.join(os.path.abspath('UMLS_KG'), 'embeddings/distmult/DTE_to_BERT.pkl'))

# DTE_Model_Lookup_Table = pd.read_pickle('DTE_to_RoBERTa.pkl')
Metamap_Tokenizations = pd.read_pickle('Metamap_Tokenizations.pkl')

# config = pickle.load(open(f"{path}/params.pkl", "rb"))
DTE_Model_Lookup_Table = pickle.load(open('DTE_to_navteca_roberta-base-squad2.pkl', 'rb'))
print('DTE_Model_Lookup_Table:\n{}'.format(DTE_Model_Lookup_Table))

model_name = 'navteca/roberta-base-squad2'

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_embeddings = model.get_input_embeddings()
n_contextual_embds = model_embeddings.weight.shape[0]

CLS_embedding = model_embeddings(torch.LongTensor([tokenizer.cls_token_id]))
SEP_embedding = model_embeddings(torch.LongTensor([tokenizer.sep_token_id]))

# all_entities = DTE_Model_Lookup_Table['Term'].to_list()
all_entities = DTE_Model_Lookup_Table['Entity'].to_list()


def custom_input_rep(ques, context, max_length=512, concat=False):
    ques = re.sub(' +', ' ', ques).strip()

    def clean_term(word):
        return re.sub(r'[\W\s]', '', word).lower()

    tup = Metamap_Tokenizations.query("Question==@ques")
    # print('ques: \"{}\"'.format(ques))
    # print('context: \"{}\"'.format(context))
    # print('tup: {}'.format(tup))  # debugging
    # print('tup columns: {}'.format(tup.columns))
    # print('tup[\'Tokenization\']: {}'.format(tup['Tokenization']))

    metamap_tokenized_question = tup['Tokenization'].values[0]
    n_original_tokens = len(metamap_tokenized_question)
    n_dte_hits = 0

    # Removing punctuations/spaces from domain-terms for easy comparison
    mappings = tup['Mappings'].values[0]
    for i, x in enumerate(mappings):
        mappings[i][0] = clean_term(x[0])

    domain_terms = [x[0] for x in mappings]

    question_embeddings = []
    input_ids = [tokenizer.cls_token_id]
    new_question_text = []
    for word in metamap_tokenized_question:
        '''
        This is done to easily check if the current word is a DT or not since DT form of the same word 
        are obtained bit differently.
        '''
        filtered_word = clean_term(word)

        '''
        This means that the filtered_word has to be a domain term which also has a KG expansion. If if does not,
        then use its BERT embeddings.
        '''

        # if filtered_word in domain_terms:  # Use DTE_BERT_Matrix
        #     mapped_concept = mappings[domain_terms.index(filtered_word)][2]
        #     print('** FIRST CONDITION SATISFIED filtered_word: {} **'.format(filtered_word))
        #     if mapped_concept in all_entities:
        #         print('** SECOND CONDITION SATISFIED **')
        #         question_embeddings.append(DTE_Model_Lookup_Table.query("Entity==@mapped_concept")['Embedding'].values[0])
        #         input_ids.append(n_contextual_embds + all_entities.index(mapped_concept))
        #
        #     new_question_text.append('a')

        if filtered_word in domain_terms and mappings[domain_terms.index(filtered_word)][2] in all_entities:  # Use DTE_BERT_Matrix
            mapped_concept = mappings[domain_terms.index(filtered_word)][2]
            question_embeddings.append(DTE_Model_Lookup_Table.query("Entity==@mapped_concept")['Embedding'].values[0].to('cpu'))
            custom_input_id = n_contextual_embds + all_entities.index(mapped_concept)
            print('filtered_word: {} concept: {} custom ID: {}'.format(filtered_word, mapped_concept, custom_input_id))
            input_ids.append(custom_input_id)
            new_question_text.append('a')

            n_dte_hits += 1
            # print('DTE HIT!')

            if concat:
                # print('$$ concatenating KGE $$')
                subword_indices = tokenizer(word)['input_ids'][1:-1]  # Take all tokens between [CLS] & [SEP]
                input_ids.extend(subword_indices)
                for index in subword_indices:
                    question_embeddings.append(model_embeddings(torch.LongTensor([index])))

                new_question_text.append(word)

        # The mapped_concept doesn't have an expansion in the KG or the term isn't a DT. Thus, its BERT embeddings are used.
        else:
            subword_indices = tokenizer(word)['input_ids'][1:-1]  # Take all tokens between [CLS] & [SEP]
            print('word: {} subword_indices: {}'.format(word, subword_indices))
            input_ids.extend(subword_indices)
            for index in subword_indices:
                question_embeddings.append(model_embeddings(torch.LongTensor([index])))

            new_question_text.append(word)

    new_question_text = ' '.join(new_question_text)

    if concat:
        effective_max_length = max_length + 15
    else:
        effective_max_length = max_length

    # Since our total i/p's can only be 512 tokens long, the context has to be adjusted accordingly.
    len_custom_question = len(question_embeddings)
    # max_length = 512
    limit_for_context = effective_max_length - (len_custom_question + 3)  # 2 to account for [CLS] & [SEP]

    context_embeddings = []

    # Taking all tokens b/w 1 & limit_for_context
    reduced_context_indices = tokenizer(context, truncation=True)['input_ids'][1:limit_for_context + 1]
    input_ids.append(tokenizer.sep_token_id)
    input_ids.extend(reduced_context_indices)
    for index in reduced_context_indices:
        context_embeddings.append(model_embeddings(torch.LongTensor([index])))
    input_ids.append(tokenizer.sep_token_id)

    # In this way, I don't have to add the CLS & SEP embeddings during fine-tuning.
    # final_representation = torch.unsqueeze(torch.cat((CLS_embedding,\
    #                                                   torch.cat([*question_embeddings]),\
    #                                                   torch.cat([*context_embeddings]),\
    #                                                   SEP_embedding)), dim=1)

    final_representation = torch.cat((CLS_embedding,
                                      torch.cat([*question_embeddings]),
                                      SEP_embedding,
                                      torch.cat([*context_embeddings]),
                                      SEP_embedding))

    n_pad = effective_max_length - final_representation.shape[0]
    attn_mask = torch.ones((effective_max_length, effective_max_length))
    for mask_idx in range(n_pad):
        attn_mask[-(mask_idx + 1), :] = 0
        attn_mask[:, -(mask_idx + 1)] = 0

    if n_pad > 0:
        model_dim = final_representation.shape[-1]
        new_padding = torch.zeros((n_pad, model_dim))
        final_representation = torch.cat([final_representation, new_padding], dim=0)

    og_input_ids = tokenizer(ques)['input_ids']
    print('og_input_ids: {}\ninput_ids: {}'.format(og_input_ids, input_ids))

    while len(input_ids) < effective_max_length:
        input_ids.append(tokenizer.pad_token_id)
    # print('!! attn_mask: {} !!'.format(attn_mask.shape))
    # print('!! final_representation: {} !!'.format(final_representation.shape))

    # This difference will be used to adjust the start/end indices of the answers in context.

    # print('input_ids: {}'.format(input_ids))
    token_diff = len(og_input_ids) - len(question_embeddings)
    input_ids = torch.tensor(input_ids)

    return final_representation, token_diff, attn_mask, new_question_text, input_ids, n_original_tokens, n_dte_hits
