import torch.nn as nn
import torch.nn.functional as F


class DenseNN(nn.Module):

    '''
    Define a Dense Neural Network that takes
    a 1d vector of RMS commands and gives in output 2 parameters: seeing and L0
    '''

    def __init__(self,neurons_first_hidden_layers):
        super().__init__()

        num_neurons = neurons_first_hidden_layers
        
        self.fc1 = nn.Linear(3, num_neurons)
        # self.fc1 = nn.Linear(400, num_neurons)
        self.fc2 = nn.Linear(num_neurons, 16)
        self.fc3 = nn.Linear(16, 6)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


