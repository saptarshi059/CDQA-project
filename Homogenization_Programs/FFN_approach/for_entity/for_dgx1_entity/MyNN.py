import torch

class FFNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim):
        super().__init__()
        self.custom_nn = torch.nn.ModuleList()
        for i in range(num_hidden_layers):
            self.custom_nn.add_module(f'Dropout{i}', torch.nn.Dropout(p=0.25))
            self.custom_nn.add_module(f'LL{i}', torch.nn.Linear(input_dim, hidden_dim))
            self.custom_nn.add_module(f'LN{i}', torch.nn.LayerNorm(hidden_dim))
            self.custom_nn.add_module(f'activation{i}', torch.nn.ReLU())
            input_dim = hidden_dim
        self.custom_nn.add_module(f'Output Layer',
                                  torch.nn.Linear(hidden_dim, output_dim))
        #self.custom_nn.add_module(f'Softmax', torch.nn.Softmax(dim=1))

    def forward(self, input_data):
        hidden_states = []
        for layer in self.custom_nn:
            output = layer(input_data)
            hidden_states.append(output)
            input_data = output
        '''
        hidden_states[-1] is the softmax o/p which we need to train the n/w with.
        hidden_states[-3] will be used during inference time to obtain the equivalent BERT variant embedding. 
        '''
        return hidden_states[-1], hidden_states[-1]