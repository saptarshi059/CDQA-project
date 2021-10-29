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

class FFN_Data(Dataset):
    def __init__(self):
        #data = pd.read_pickle('Entity_Homogenization_data.pkl')
        data = pickle.load(open('Entity_Homogenization_data.pkl', 'rb'))
        self.x = data['train']
        self.y = data['test']
        self.n_samples = data.shape[0]
    
    def __getitem__(self, index):
        #true_output = np.zeros(vocab_size)
        #np.put(true_output, self.y[index], 1) #Creating true target representation here
        #return self.x[index], true_output
        return torch.FloatTensor(self.x[index]), torch.FloatTensor(self.y[index])
    
    def __len__(self):
        return self.n_samples

homogenization_dataset = FFN_Data()
dataloader = DataLoader(dataset=homogenization_dataset, batch_size=64, shuffle=True, num_workers=2)

print('Dataloader object created...')

#BERT_variant = 'phiyodr/bert-base-finetuned-squad2'

#ent_embeddings_size = len(pd.read_csv(os.path.join(os.path.join(os.getcwd(), os.path.relpath('embeddings/distmult')), 'ent_embedding.tsv'), sep='\t', header=None).columns)
#input_dimension = ent_embeddings_size

input_dimension = homogenization_dataset.x[0].shape[0]

#output_dimension = vocab_size
output_dimension = homogenization_dataset.y[0].shape[0]

#Play with this
number_of_hidden_layers = 5

#Size of embedding required by BERT variant (usually 768)
hidden_dimension = 768

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

num_epochs = 100
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
        softmax_output, _ = model(train_sample.to(device))       
               
        #Compute Loss
        loss = criterion(softmax_output.float().to(device), true_output.float().to(device))

        #Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if (iteration_num + 1) % 5 == 0:
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