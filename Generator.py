import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Generator(nn.Module):

    def __init__(self, input_dim, action_space_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(100, 100)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(100, action_space_dim)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.logsoftmax = nn.LogSoftmax()


    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.logsoftmax(x)