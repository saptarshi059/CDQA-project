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
        self.use_kge = getattr(self.args, 'use_kge', False)
        self.use_dict = getattr(self.args, 'use_dict', False)
        self.concat_kge = getattr(self.args, 'concat_kge', False)
        self.fancy_concat = getattr(self.args, 'fancy_concat', False)
        assert not (self.concat_kge and self.fancy_concat), 'Can only select one of concat_kge or fancy_concat'

        print('InputMaker reading metamap...')
        self.metamap_tokenizations = pickle.load(open(self.metamap_tokenizations_pkl_fp, 'rb'))

        print('InputMaker reading DTE lookup table...')
        self.DTE_model_lookup_table = pickle.load(open(self.dte_lookup_table_fp, 'rb'))
        self.all_entities = self.DTE_model_lookup_table['Entity'].tolist()

        print('InputMaker creating model and tokenizer...')
        print('\tmodel name: {}'.format(self.model_name))
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.sep_token = self.tokenizer.sep_token
        self.pad_id = self.tokenizer.pad_token_id
        self.n_contextual_embds = self.model.get_input_embeddings().weight.shape[0]

    def convert_questions_to_kge(self, q_text):
        # print('** raw q_text: {} **'.format(q_text))
        q_text = re.sub(' +', ' ', q_text).strip()
        q_text = '{} '.format(q_text)
        # print('** q_text: \'{}\' **'.format(q_text))
        tup = self.metamap_tokenizations.query("Question==@q_text")

        if not tup.empty:
            mappings = tup['Mappings'].values[0]
            if self.fancy_concat:
                new_text_components = [q_text.strip()]
                domain_umls_terms = ' '.join(['[{}]'.format(m[2]) for m in mappings])
                domain_dict_terms = ' '.join(['#{}#'.format(m[2]) for m in mappings])
                if self.use_kge:
                    new_text_components.extend([self.sep_token, domain_umls_terms])
                if self.use_dict:
                    new_text_components.extend([self.sep_token, domain_dict_terms])
                q_text = ' '.join(new_text_components)

            else:
                mappings = list(sorted(mappings, key=lambda x: len(x[0])))

                for text_str, _, domain_term in mappings:
                    q_text = self.add_kge_to_text(q_text, text_str, domain_term)
                # print('** new q_text: {} **'.format(q_text))
        else:
            print('Question \'{}\' does not have mappings!'.format(q_text))
        return q_text

    def add_kge_to_text(self, q_text, text_to_match, domain_term):
        text_components = []
        curr_text = q_text

        while text_to_match in curr_text:
            dt_index = curr_text.index(text_to_match)
            prefix = curr_text[:dt_index].strip()
            text_components.append(prefix)

            if self.concat_kge:
                kge_text = '{} / [{}]'.format(text_to_match, domain_term)
            else:
                kge_text = '[{}]'.format(domain_term)
            text_components.append(kge_text)

            curr_text = curr_text[dt_index + len(text_to_match):].strip()
        text_components.append(curr_text)

        # print('text_components: {}'.format(text_components))
        new_text = ' '.join(text_components)
        # input('new_text: {}'.format(new_text))
        return new_text





