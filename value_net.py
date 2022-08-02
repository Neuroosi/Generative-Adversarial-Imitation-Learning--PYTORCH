import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class value_net(nn.Module):

    def __init__(self, input_dim):
        super(value_net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        #torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        #torch.nn.init.kaiming_uniform_(self.fc2.weight)
        #torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.logsoftmax = nn.LogSoftmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.value = nn.Linear(50,1)

    def forward(self, x):
        x = x.to(device)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.value(x)
        return x