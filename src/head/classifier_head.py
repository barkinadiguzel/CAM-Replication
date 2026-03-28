import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, x):
        x = self.fc(x)      
        x = self.softmax(x) 
        return x
