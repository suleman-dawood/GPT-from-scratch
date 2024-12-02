import torch.nn as nn

class FeedForward(nn.Module): # simple neural network

    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, num_embeddings),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)