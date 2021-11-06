#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Creating the feedforward network
import torch
import pandas as pd
import os
from transformers import AutoTokenizer
from MyNN import FFNN
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle5 as pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('-hl', '--num_hidden_layers', default=5, type=int)
parser.add_argument('-hd', '--hidden_dim', default=300, type=int)
parser.add_argument('-e', '--epochs', default=300, type=int)

args = parser.parse_args()

class FFN_Data(Dataset):
    def __init__(self):
        data = pickle.load(open('training_data_Homogenization_data.pkl', 'rb'))
        self.x = data['train']
        self.y = data['test']
        self.n_samples = data.shape[0]
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.x[index]), torch.FloatTensor(self.y[index])
    
    def __len__(self):
        return self.n_samples

homogenization_dataset = FFN_Data()
dataloader = DataLoader(dataset=homogenization_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

print('Dataloader object created...')

input_dimension = homogenization_dataset.x[0].shape[0]

output_dimension = homogenization_dataset.y[0].shape[0]

number_of_hidden_layers = args.num_hidden_layers

hidden_dimension = args.hidden_dim

#Initializing the network
model = FFNN(input_dimension, output_dimension, number_of_hidden_layers, hidden_dimension)

#We have to make send the model to device before creating the optimizer since parameters of a model after 
#.cuda() will be different objects with those before the call (https://pytorch.org/docs/stable/optim.html)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.MSELoss()

#Adam optimizer
optimizer = torch.optim.Adam(model.parameters())

print(f'FFN architecture: \n {model} \n Loss Function: {criterion} \n Optimizer: {optimizer}')
print(f'Device being used: {device}')

#Training Loop

num_epochs = args.epochs
num_iterations = np.ceil(len(homogenization_dataset)/dataloader.batch_size)

#Setting model to train mode
model.train()
epoch_loss_list = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for iteration_num, (train_sample, true_output) in enumerate(dataloader):

        #Clearing Gradients
        optimizer.zero_grad()
        
        #Generating Predictions
        #'_' for hidden_states[-3] since we don't need that for training
        activation_output, _ = model(train_sample.to(device))       
               
        #Compute Loss
        loss = criterion(activation_output.float().to(device), true_output.float().to(device))

        #Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if (iteration_num + 1) % 1000 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs} | Iteration: {iteration_num+1}/{num_iterations} | Current Loss: {loss.item()}')
    
    epoch_loss_list.append(epoch_loss)
    print(f'Epoch: {epoch+1}/{num_epochs} | Epoch Loss {epoch_loss}')

print('Saving model to disk...')
torch.save(model.state_dict(), 'Homogenizer.pt')
print('Model saved...')

plt.plot(list(range(len(epoch_loss_list))), epoch_loss_list)
plt.xlabel('epoch number')
plt.ylabel('epoch loss')
plt.savefig('Entity_Training_Loss_Plot.png', bbox_inches='tight', dpi=600)