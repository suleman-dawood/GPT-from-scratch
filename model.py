import torch.nn as nn
from torch.nn import functional
import torch

class BigramModel(nn.Module):

    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)
        self.loss_fn = nn.CrossEntropyLoss()

    # forward pass
    def forward(self, index, targets=None):
        logits = self.embedding_table(index)

        if targets==None:
            loss = None
        else:
            # logits and targets must be resized to be used with NNs
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = self.loss_fn(logits, targets)

        return logits, loss

    # generate likely tokens that come after selected index
    def generate(self, index, new_tokens):

        for i in range(new_tokens):
            logits, loss = self(index)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1) # apply softmax to get probability distribution
            next_index = torch.multinomial(probs, num_samples=1) # retrieve sample from the distribution
            index = torch.cat((index, next_index), dim=1) # # append sample  to the full sequence
        return index