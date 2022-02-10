__author__ = 'Connor Heaton and Saptarshi Sengupta'

import json
import torch
import pickle5 as pickle

from input_maker import InputMaker
from torch.utils.data import Dataset


class CovidQADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) if key not in ['question_texts', 'context_texts'] else val[idx] for key, val
                in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def load_pubmed_data(data_fp, maker=None):
    data = []
    j = json.load(open(data_fp))
    label_map = {'yes': 0, 'no': 1, 'maybe': 2}

    for q_id, q_d in j.items():
        question = q_d['QUESTION']
        if maker is not None:
            question = maker.convert_questions_to_kge(question)

        labels = q_d['LABELS']
        contexts = q_d['CONTEXTS']
        context = ['{}: {}'.format(label_.capitalize(), context_) for label_, context_ in zip(labels, contexts)]
        context = ' '.join(context)
        label = label_map[q_d['final_decision']]
        this_item = {
            'id': q_id,
            'question': question,
            'context': context,
            'label': label
        }
        data.append(this_item)

    return data


class PubmedQADataset(Dataset):
    def __init__(self, args, data_fp, tokenizer):
        self.args = args
        self.data_fp = data_fp
        self.tokenizer = tokenizer

        self.max_seq_len = self.args.max_seq_len
        self.use_kge = self.args.use_kge
        if self.use_kge:
            # In dataset, we just need to adjust question text, tokenizer already updated
            my_maker = InputMaker(args)
        else:
            my_maker = None

        self.items = load_pubmed_data(self.data_fp, maker=my_maker)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_data = self.items[idx]

        item_id = int(item_data['id'])
        question_str = item_data['question']
        context_str = item_data['context']
        label = item_data['label']

        embds = self.tokenizer.encode_plus(
            text=question_str,
            text_pair=context_str,
            max_length=self.max_seq_len,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation='longest_first',
            return_tensors='pt'
        )
        item_id = torch.tensor([item_id])
        label = torch.tensor([label])

        out = {
            'item_id': item_id,
            'input_ids': embds['input_ids'],
            'attention_mask': embds['attention_mask'],
            'token_type_ids': embds['token_type_ids'],
            'label': label,
            'question_str': question_str,
            'context_str': context_str,
        }

        return out
