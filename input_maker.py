__author__ = 'Connor Heaton and Saptarshi Sengupta'

import re
import torch

import pandas as pd
import pickle5 as pickle

from transformers import AutoTokenizer, AutoModel


def clean_term(word):
    return re.sub(r'[\W\s]', '', word).lower()


class InputMaker(object):
    def __init__(self, args):
        self.args = args

        self.metamap_tokenizations_pkl_fp = getattr(self.args,
                                                    'metamap_tokenizations_pkl_fp',
                                                    'Metamap_Tokenizations.pkl')
        self.dte_lookup_table_fp = getattr(self.args,
                                           'dte_lookup_table_fp',
                                           'DTE_to_phiyodr_bert-base-finetuned-squad2.pkl')
        self.model_name = getattr(self.args, 'model_name', 'bert-base-uncased')
        self.max_len = getattr(self.args, 'max_len', 384)
        self.n_stride = getattr(self.args, 'n_stride', 164)
        self.concat_kge = getattr(self.args, 'concat_kge', False)

        print('InputMaker reading metamap...')
        self.metamap_tokenizations = pd.read_pickle(self.metamap_tokenizations_pkl_fp)

        print('InputMaker reading DTE lookup table...')
        self.DTE_model_lookup_table = pickle.load(open(self.dte_lookup_table_fp, 'rb'))
        self.all_entities = self.DTE_model_lookup_table['Entity'].tolist()

        print('InputMaker creating model and tokenizer...')
        print('\tmodel name: {}'.format(self.model_name))
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.n_contextual_embds = self.model.get_input_embeddings().weight.shape[0]

    def make_inputs(self, ques, context, verbosity=0):
        ques = re.sub(' +', ' ', ques).strip()
        tup = self.metamap_tokenizations.query("Question==@ques")
        if verbosity > 0:
            print('tup: {}'.format(tup))
        metamap_tokenized_question = tup['Tokenization'].values[0]
        n_original_tokens = len(metamap_tokenized_question)
        n_dte_hits = 0

        # Removing punctuations/spaces from domain-terms for easy comparison
        mappings = tup['Mappings'].values[0]
        if verbosity > 0:
            print('mappings: {}'.format(mappings))

        for i, x in enumerate(mappings):
            mappings[i][0] = clean_term(x[0])

        domain_terms = [x[0] for x in mappings]

        question_embeddings = []
        input_ids = [self.cls_id]
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
            if filtered_word in domain_terms and mappings[domain_terms.index(filtered_word)][2] in self.all_entities:
                mapped_concept = mappings[domain_terms.index(filtered_word)][2]
                # question_embeddings.append(
                #     self.DTE_model_lookup_table.query("Entity==@mapped_concept")['Embedding'].values[0].to('cpu'))
                custom_input_id = self.n_contextual_embds + self.all_entities.index(mapped_concept)
                if verbosity > 0:
                    print('filtered_word: \"{}\"\tconcept: \"{}\"\tcustom ID: {}'.format(filtered_word, mapped_concept,
                                                                                     custom_input_id))
                input_ids.append(custom_input_id)
                new_question_text.append('a')

                n_dte_hits += 1
                # print('DTE HIT!')

                if self.concat_kge:
                    # print('$$ concatenating KGE $$')
                    subword_indices = self.tokenizer(word)['input_ids'][1:-1]  # Take all tokens between [CLS] & [SEP]
                    input_ids.extend(subword_indices)
                    # for index in subword_indices:
                    #     question_embeddings.append(model_embeddings(torch.LongTensor([index])))

                    new_question_text.append(word)

            # The mapped_concept doesn't have an expansion in the KG or the term isn't a DT. Thus, its BERT embeddings are used.
            else:
                subword_indices = self.tokenizer(word)['input_ids'][1:-1]  # Take all tokens between [CLS] & [SEP]
                if verbosity > 0:
                    print('word: \"{}\"\tsubword_indices: {}'.format(word, subword_indices))
                input_ids.extend(subword_indices)
                # for index in subword_indices:
                #     question_embeddings.append(model_embeddings(torch.LongTensor([index])))

                new_question_text.append(word)

        new_question_text = ' '.join(new_question_text)

        if self.concat_kge:
            effective_max_length = self.max_len + 15
        else:
            effective_max_length = self.max_len

        # Since our total i/p's can only be 512 tokens long, the context has to be adjusted accordingly.
        len_custom_question = len(question_embeddings)
        # max_length = 512
        limit_for_context = effective_max_length - (len_custom_question + 3)  # 2 to account for [CLS] & [SEP]

        context_embeddings = []
        input_ids.append(self.sep_id)

        # Taking all tokens b/w 1 & limit_for_context
        # reduced_context_indices = self.tokenizer(context, truncation=True)['input_ids'][1:limit_for_context + 1]
        reduced_context_indices = self.tokenizer(context, truncation=True, max_length=self.max_len)['input_ids'][1:-1]
        input_ids.extend(reduced_context_indices)
        if len(input_ids) >= self.max_len:
            input_ids = input_ids[:self.max_len - 1]

        # for index in reduced_context_indices:
        #     context_embeddings.append(model_embeddings(torch.LongTensor([index])))
        input_ids.append(self.sep_id)
        og_input_ids = self.tokenizer(ques, context)['input_ids']
        if verbosity > 0:
            print('Raw question: {}'.format(ques))
            print('og_input_ids: {}\ncustom input_ids: {}'.format(og_input_ids, input_ids))
        token_diff = len(og_input_ids) - len(question_embeddings)
        n_pad = effective_max_length - len(input_ids)
        attn_mask = torch.ones((effective_max_length, effective_max_length))
        for mask_idx in range(n_pad):
            attn_mask[-(mask_idx + 1), :] = 0
            attn_mask[:, -(mask_idx + 1)] = 0

        while len(input_ids) < effective_max_length:
            input_ids.append(self.pad_id)

        input_ids = torch.tensor(input_ids)

        return token_diff, attn_mask, new_question_text, input_ids, n_original_tokens, n_dte_hits

    def make_pipeline_inputs(self, ques, context, verbosity=0):
        print('ques: \"{}\"'.format(ques))
        print('context: \"{}\"'.format(context))

        ques = re.sub(' +', ' ', ques).strip()
        tup = self.metamap_tokenizations.query("Question==@ques")
        if verbosity > 0:
            print('tup: {}'.format(tup))
        metamap_tokenized_question = tup['Tokenization'].values[0]

        # Removing punctuations/spaces from domain-terms for easy comparison
        mappings = tup['Mappings'].values[0]
        if verbosity > 0:
            print('mappings: {}'.format(mappings))

        for i, x in enumerate(mappings):
            mappings[i][0] = clean_term(x[0])

        domain_terms = [x[0] for x in mappings]
        q_input_ids = [self.cls_id]
        new_question_text = []

        print('Iterating over question words...')
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
            if filtered_word in domain_terms and mappings[domain_terms.index(filtered_word)][2] in self.all_entities:
                mapped_concept = mappings[domain_terms.index(filtered_word)][2]
                custom_input_id = self.n_contextual_embds + self.all_entities.index(mapped_concept)
                if verbosity > 0:
                    print('filtered_word: \"{}\"\tconcept: \"{}\"\tcustom ID: {}'.format(filtered_word, mapped_concept,
                                                                                         custom_input_id))
                q_input_ids.append(custom_input_id)
                new_question_text.append('a')

                if self.concat_kge:
                    if verbosity > 0:
                        print('$$ concatenating KGE $$')
                    subword_indices = self.tokenizer(word)['input_ids'][1:-1]  # Take all tokens between [CLS] & [SEP]
                    q_input_ids.extend(subword_indices)

                    new_question_text.append(word)
            else:
                subword_indices = self.tokenizer(word)['input_ids'][1:-1]  # Take all tokens between [CLS] & [SEP]
                if verbosity > 0:
                    print('word: \"{}\"\tsubword_indices: {}'.format(word, subword_indices))
                q_input_ids.extend(subword_indices)
                # for index in subword_indices:
                #     question_embeddings.append(model_embeddings(torch.LongTensor([index])))

                new_question_text.append(word)

        print('Iterating over context w/ stride...')
        new_question_text = ' '.join(new_question_text)
        q_input_ids.append(self.sep_id)
        # context_ids = self.tokenizer(context, truncation=True, max_length=self.max_len,  stride=self.n_stride)['input_ids'][1:-1]
        context_ids = self.tokenizer(context, max_length=9999999)['input_ids']
        print('context_ids: {}'.format(len(context_ids)))
        n_context_ids_for_sample = self.max_len - len(q_input_ids) - 1
        print('n_context_ids_for_sample: {}'.format(n_context_ids_for_sample))
        comb_input_ids = []
        comb_attn_masks = []
        curr_context_start_idx = 0
        while curr_context_start_idx < len(context_ids):
            these_input_ids = q_input_ids[:]
            these_input_ids.extend(context_ids[curr_context_start_idx:curr_context_start_idx+n_context_ids_for_sample])
            these_input_ids.append(self.sep_id)
            n_pad = self.max_len - len(these_input_ids)
            print('n_pad: {}'.format(n_pad))
            curr_context_start_idx += n_context_ids_for_sample - self.n_stride

            while len(these_input_ids) < self.max_len:
                these_input_ids.append(self.pad_id)

            these_input_ids = torch.tensor(these_input_ids).unsqueeze(0)

            this_attn_mask = torch.ones((self.max_len, self.max_len))
            for mask_idx in range(n_pad):
                this_attn_mask[-(mask_idx + 1), :] = 0
                this_attn_mask[:, -(mask_idx + 1)] = 0

            comb_input_ids.append(these_input_ids)
            comb_attn_masks.append(this_attn_mask.unsqueeze(0))

        comb_input_ids = torch.cat(comb_input_ids, dim=0)
        comb_attn_masks = torch.cat(comb_attn_masks, dim=0)

        return comb_input_ids, comb_attn_masks
