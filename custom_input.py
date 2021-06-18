#Extracting embeddings for non-domain terms. I'm simply using BERT's tokenizer for the nDT's.
#Creating question representations in this block.
from transformers import AutoTokenizer, AutoModel
import torch
import re
import os
import pandas as pd

#DTE_Model_Lookup_Table = pd.read_pickle(os.path.join(os.path.abspath('UMLS_KG'), 'embeddings/distmult/DTE_to_BERT.pkl'))

DTE_Model_Lookup_Table = pd.read_pickle('DTE_to_RoBERTa.pkl')
Metamap_Tokenizations = pd.read_pickle('Metamap_Tokenizations.pkl')

model_name = 'navteca/roberta-base-squad2'

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_embeddings = model.get_input_embeddings()

CLS_embedding = model_embeddings(torch.LongTensor([tokenizer.cls_token_id]))
SEP_embedding = model_embeddings(torch.LongTensor([tokenizer.sep_token_id]))

all_entities = DTE_Model_Lookup_Table['Term'].to_list()

def custom_input_rep(ques, context):
    
    def clean_term(word):
        return re.sub(r'[\W\s]', '', word).lower()

    tup = Metamap_Tokenizations.query("Question==@ques")
    print('ques: \"{}\"'.format(ques))
    print('context: \"{}\"'.format(context))
    print('tup: {}'.format(tup))    # debugging
    print('tup columns: {}'.format(tup.columns))
    print('tup[\'Tokenization\']: {}'.format(tup['Tokenization']))
    
    metamap_tokenized_question = tup['Tokenization'].values[0]

    #Removing punctuations/spaces from domain-terms for easy comparison
    mappings = tup['Mappings'].values[0]
    for i,x in enumerate(mappings):
        mappings[i][0] = clean_term(x[0])

    domain_terms = [x[0] for x in mappings]

    question_embeddings = []
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

        if filtered_word in domain_terms: #Use DTE_BERT_Matrix
            mapped_concept = mappings[domain_terms.index(filtered_word)][1]
            if mapped_concept in all_entities: 
                question_embeddings.append(DTE_Model_Lookup_Table.query("Term==@mapped_concept")['Embedding'].values[0])
            
        #The mapped_concept doesn't have an expansion in the KG or the term isn't a DT. Thus, its BERT embeddings are used.
        else:
            subword_indices = tokenizer(word)['input_ids'][1:-1] #Take all tokens between [CLS] & [SEP]
            for index in subword_indices:
                question_embeddings.append(model_embeddings(torch.LongTensor([index])))
    
    #Since our total i/p's can only be 512 tokens long, the context has to be adjusted accordingly.
    len_custom_question = len(question_embeddings)
    max_length = 512
    limit_for_context = max_length - (len_custom_question + 2) #2 to account for [CLS] & [SEP]
    
    context_embeddings = []
    
    #Taking all tokens b/w 1 & limit_for_context
    reduced_context_indices = tokenizer(context, truncation=True)['input_ids'][1:limit_for_context+1]
    
    for index in reduced_context_indices:
        context_embeddings.append(model_embeddings(torch.LongTensor([index])))
        
    #In this way, I don't have to add the CLS & SEP embeddings during fine-tuning.
    final_representation = torch.unsqueeze(torch.cat((CLS_embedding,\
                                                      torch.cat([*question_embeddings]),\
                                                      torch.cat([*context_embeddings]),\
                                                      SEP_embedding)), dim=1)
    
    #This difference will be used to adjust the start/end indices of the answers in context.
    token_diff = len(tokenizer(ques)['input_ids']) - len(question_embeddings)
       
    return final_representation, token_diff