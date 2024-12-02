import torch.nn as nn
from multihead import *
from feedforward import *

class Block(nn.Module):
    # transformer block that contains all internal mechanisms
    def __init__(self, num_embeddings, head_count):
        super().__init__()
        head_size = num_embeddings // head_count
        self.head = MultiHead(head_count, head_size) # feature extraction layer
        self.feed_forward = FeedForward(num_embeddings) # feedforward layer to add non-linearity
        self.ln1 = nn.LayerNorm(num_embeddings) # normalises features
        self.ln2 = nn.LayerNorm(num_embeddings)

    def forward(self, x):
        x = x + self.head(self.ln1(x)) # using += for optimization using residual connections
        x = x + self.feed_forward(self.ln2(x))
        return x
