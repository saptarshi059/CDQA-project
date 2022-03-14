'''
The same code is there in the last 2 cells of KG_Constructor. Brought them here for clarity.
'''

#python CUI_PC+MM_Tokenizations_gen.py -mmf MM_New.json

#Generate tuples (question, (matched_text, CUI, preferred candidate)) - new approach
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mmf', '--MetaMap_Output', type=str)
args = parser.parse_args()

texts = json.load(open(args.MetaMap_Output))['AllDocuments'][0]['Document']['Utterances']

Metamap_Tokenizations = []
for ques in texts:
    mappings = []
    question_text = ques['UttText']
    ques_start_idx = int(ques['UttStartPos'])
    for phr in ques['Phrases']:
        if phr['Mappings'] != []:
            for phr_dict in phr["Mappings"][0]['MappingCandidates']: #Choosing the first candidate
                start_idx = int(phr_dict['ConceptPIs'][0]['StartPos']) - ques_start_idx
                end_idx = start_idx + int(phr_dict['ConceptPIs'][0]['Length'])
                mappings.append((question_text[start_idx:end_idx], phr_dict['CandidateCUI'], \
                                 phr_dict['CandidatePreferred']))
    Metamap_Tokenizations.append((question_text, mappings))

entities = set()
for mappings in Metamap_Tokenizations:
    for tup in mappings[1]:
        entities.add(tup[2])
print(f"Number of entities discovered: {len(entities)}")

#Saving Metamap_Tokenizations for use during question embedding creation
pd.DataFrame(Metamap_Tokenizations, columns=['Question', 'Mappings']).to_pickle('Metamap_Tokenizations.pkl')
print('Tokenizations table generated...')

#New version of CUI-PC_Lookup table
cuis = [y[1] for x in Metamap_Tokenizations for y in x[1]]
pc = [y[2] for x in Metamap_Tokenizations for y in x[1]]
CUI_Preferred_Concept_Lookup_Table = pd.DataFrame(zip(cuis, pc), columns=['CUI','Preferred_Concept']).drop_duplicates()
CUI_Preferred_Concept_Lookup_Table.to_csv('CUI_PC.csv', index=False)
print('Our_CUI_PC table generated...')