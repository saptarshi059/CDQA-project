#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing necessary files
import torch
import pandas as pd
import os
from transformers import AutoTokenizer
from MyNN import FFNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-BV','--BERT_variant', default='navteca/roberta-base-squad2')
parser.add_argument('-f', '--folds', default=5, type=int)
parser.add_argument('-e', '--epochs', default=50, type=int)
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('-nl', '--num_hidden_layers', default=5, type=int)
args = parser.parse_args()

BERT_variant = args.BERT_variant

vocab_size = AutoTokenizer.from_pretrained(BERT_variant).vocab_size
ent_embeddings_size = len(pd.read_csv(os.path.relpath('embeddings/distmult/ent_embedding.tsv'), sep='\t', header=None).columns)

# In[ ]:


#Creating dataset object & dataloader
from torch.utils.data import Dataset, DataLoader
import numpy as np
class FFN_Data(Dataset):
    def __init__(self):
        data = pd.read_pickle('Homogenization_data.pkl')
        self.x = data['train']
        self.y = data['test']
        self.n_samples = data.shape[0]
    
    def __getitem__(self, index):
        true_output = np.zeros(vocab_size)
        np.put(true_output, self.y[index], 1) #Creating true target representation here
        return self.x[index], true_output
    
    def __len__(self):
        return self.n_samples

homogenization_dataset = FFN_Data()
print('Dataset object created...')


# In[ ]:


'''
Code adapted from 
https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
'''
from sklearn.model_selection import KFold

def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

# Configuration options
k_folds = args.folds
num_epochs = args.epochs
loss_function = torch.nn.BCELoss()
batch_size = args.batch_size
#1 X [dim of 1 KGE], since we are doing mean(triple)
input_dimension = ent_embeddings_size

#Size of BERT variant vocabulary
output_dimension = vocab_size

#Play with this
number_of_hidden_layers = args.num_hidden_layers

#Size of embedding required by BERT variant (usually 768)
hidden_dimension = 768
  
# For fold results
test_fold_loss = {}
train_fold_loss = {}

# Set fixed random number seed
torch.manual_seed(42)
  
# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# Start print
print('--------------------------------')

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(homogenization_dataset)):
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      homogenization_dataset, 
                      batch_size=batch_size, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      homogenization_dataset,
                      batch_size=batch_size, sampler=test_subsampler)

    # Init the neural network
    model = FFNN(input_dimension, output_dimension, number_of_hidden_layers, hidden_dimension)
    model.apply(reset_weights)
    '''
    We have to make send the model to device before creating the optimizer since parameters of a model after 
    .cuda() will be different objects with those before the call (https://pytorch.org/docs/stable/optim.html)
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Run the training loop for defined number of epochs
    epoch_loss_list = []
    for epoch in range(0, num_epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0
        epoch_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            #'_' for hidden_states[-3] since we don't need that for training
            softmax_output, _ = model(inputs.to(device))

            #Compute Loss
            loss = loss_function(softmax_output.double().to(device), targets.double().to(device))

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            epoch_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
                current_loss = 0.0
        
        epoch_loss_list.append(epoch_loss)
    
    #Average training loss over all epochs.
    train_fold_loss[fold] = sum(epoch_loss_list)/num_epochs        
    
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')

    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # Evaluation for this fold
    correct, total = 0, 0

    with torch.no_grad():
        test_loss = 0
        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, targets = data

            # Generate outputs
            softmax_output, _ = model(inputs.to(device))

            loss = loss_function(softmax_output.double().to(device), targets.double().to(device))
            
            test_loss += loss.item()

        test_fold_loss[fold] = test_loss

# Print fold results
print(f'K-FOLD CROSS VALIDATION TRAIN LOSS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in train_fold_loss.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average training loss: {sum/len(train_fold_loss.items())} %')

print(f'K-FOLD CROSS VALIDATION TEST LOSS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in test_fold_loss.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average test loss: {sum/len(test_fold_loss.items())} %')


# In[ ]:


#Plotting loss
import matplotlib.pyplot as plt

x = list(range(k_folds))

fig = plt.figure()
ax1 = fig.add_subplot(111)

# create line plot of training loss
line1, = ax1.plot(x, list(train_fold_loss.values()), 'g', label="Training Loss")
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color='g')

# create shared axis for y2(x)
ax2 = ax1.twinx()

# create line plot of y2(x)
line2, = ax2.plot(x, list(test_fold_loss.values()), 'r', label="Test Loss")
ax2.set_ylabel('Test Loss', color='r')

# set title, plot limits, etc
plt.title('Tracking Loss over K-Folds')

# add a legend, and position it on the upper right
plt.legend((line1, line2), ('Training Loss', 'Test Loss'))

plt.savefig('KFold_Loss_Plot.png', bbox_inches='tight', dpi=600)
plt.show()

