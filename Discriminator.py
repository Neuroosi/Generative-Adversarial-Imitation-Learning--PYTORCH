import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Discriminator(nn.Module):

    def __init__(self, input_dim, IS_DISCRETE):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        #torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(50, 50)
        #torch.nn.init.kaiming_uniform_(self.fc2.weight)
        self.fc4 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        #torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.logsigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.discrete = IS_DISCRETE
        
    def forward(self, x):
        x = x.to(device)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc4(x))
        x = self.fc3(x)
        return x