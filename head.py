from constants import *
import torch
import torch.nn as nn

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        # these linear layers compute the key, query and value for attention weight calculation
        self.key = nn.Linear(num_embeddings, head_size, bias = False)
        self.query = nn.Linear(num_embeddings, head_size, bias = False)
        self.value = nn.Linear(num_embeddings, head_size, bias = False)
        # this lower triangular matrix acts as a buffer to prevent future tokens from being seen
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(DROPOUT) # dropout for regularization

    # forward pass throught the attention calculations
    def forward(self, x):
        Key = self.key(x) # all tokens
        Query = self.query(x) # current token
        Value = self.value(x) # actual data used in the ouput

        scale_factor = Key.shape[-1]**-0.5 # scale to stabilize
        attention_weights = Query @ Key.transpose(-2, -1) * scale_factor # calculates how much each token relates to every other token

        mask = self.tril[:context_size, :context_size]
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        attention_weights = nn.functional.softmax(attention_weights, dim=1)
        attention_weights = self.dropout(attention_weights)
        output = attention_weights @ Value
        return output
