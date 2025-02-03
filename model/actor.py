import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor (nn.Module):
  def __init__ (self):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(10, 512) #  , 4>>10
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, 128)
    self.fc4 = nn.Linear(128, 1)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    self.init_weights()
  
  def init_weights (self):
    nn.init.kaiming_uniform_(self.fc1.weight.data)
    nn.init.constant_(self.fc1.bias.data, 0)
    nn.init.kaiming_uniform_(self.fc2.weight.data)
    nn.init.constant_(self.fc2.bias.data, 0)
    nn.init.kaiming_uniform_(self.fc3.weight.data)
    nn.init.constant_(self.fc3.bias.data, 0)
    nn.init.xavier_uniform_(self.fc4.weight.data)
    nn.init.constant_(self.fc4.bias.data, 0)
  
  def forward (self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.relu(out)
    out = self.fc3(out)
    out = self.relu(out)
    out = self.fc4(out)
    out = self.tanh(out)
    return out
