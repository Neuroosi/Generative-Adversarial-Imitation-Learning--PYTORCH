import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Generator(nn.Module):

    def __init__(self, input_dim, action_space_dim, discrete):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        #torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(100, 100)
        #torch.nn.init.kaiming_uniform_(self.fc2.weight)
        self.fc3_discrete = nn.Linear(100, action_space_dim)
        self.fc3_cts = nn.Linear(100, action_space_dim)
        #torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.logsoftmax = nn.LogSoftmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.discrete = discrete
        if discrete is False:
            self.log_std = nn.Parameter(torch.zeros(action_space_dim))
        self.value = nn.Linear(100,1)

    def forward(self, x):
        x = x.to(device)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        if self.discrete:
            x = self.fc3_discrete(x)
            return self.logsoftmax(x)
        value = self.value(x)
        x = self.fc3_cts(x)
        return x, value