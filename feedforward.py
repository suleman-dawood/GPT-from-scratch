import torch.nn as nn
from constants import *

class FeedForward(nn.Module): # simple neural network

    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)