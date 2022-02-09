#python iternorm.py pre_norm_data.pkl norm_embs.txt

from argparse import ArgumentParser
import numpy as np
import pandas as pd

def load_embed(filename, max_vocab=-1):
    #words, embeds = [], []
    data_file = pd.read_pickle(filename)
    words = data_file.PC.to_list()
    embeds = data_file.source_embedding.to_list()
    '''
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            word, vector = line.rstrip().split(' ', 1)
            vector = np.fromstring(vector, sep=' ')
            words.append(word)
            embeds.append(vector)
            if len(embeds) == max_vocab:
                break
    '''
    return words, np.array(embeds), np.array(data_file.target_embedding.to_list())
    

def main():
    parser = ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--normalize', default='renorm,center,renorm,center,renorm,center,renorm,center,renorm,center,renorm', type=str)
    parser.add_argument('--max_vocab', default=-1, type=int)
    args = parser.parse_args()

    words, embeds, target_embeds = load_embed(args.input_file, max_vocab=args.max_vocab)

    print(f"Before Normalization, Embedding [0]: {embeds[0]}")
    for t in args.normalize.split(','):
        if t == 'center':
            embeds -= embeds.mean(axis=0)[np.newaxis, :]
        elif t == 'renorm':
            embeds /= np.linalg.norm(embeds, axis=1)[:, np.newaxis] + 1e-8
        elif t != '':
            raise Exception('Unknown normalization type: "%s"' % t)

    print(f"After Normalization, Embedding [0]: {embeds[0]}")
    with open(args.output_file, 'w') as f:
        print(embeds.shape[0], embeds.shape[1], file=f)
        for word, embed in zip(words, embeds):
            vector_str = ' '.join(str(x) for x in embed) #this line was fucked up - just check the original code.
            print(word.replace(' ', '_'), vector_str, file=f)
    print('Normalized UMLS embeddings generated...')

    with open('target_embeds.txt', 'w') as f:
        print(target_embeds.shape[0], target_embeds.shape[1], file=f)
        for word, embed in zip(words, target_embeds):
            vector_str = ' '.join(str(x) for x in embed) #this line was fucked up - just check the original code.
            print(word.replace(' ', '_'), vector_str, file=f)
    print('Target embeddings saved in txt format...')

if __name__ == '__main__':
    main()
