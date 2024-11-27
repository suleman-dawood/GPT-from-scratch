import torch.nn as nn
from torch.nn import functional
import torch
from constants import *
from head import *

class BigramModel(nn.Module):

    def __init__(self):
        super().__init__()
        # converts token into dense vectors which capture semantic info/ context about the word
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, num_embeddings)
        # converts each position in the input sequence into dense vectore which captures info about sequence of tokens
        self.position_embedding_table = nn.Embedding(context_size, num_embeddings)
        # create head
        self.head = Head(num_embeddings)
        # converts to logits
        self.linear_projection = nn.Linear(num_embeddings, VOCAB_SIZE)
        # loss function to compute gradients
        self.loss_fn = nn.CrossEntropyLoss()

    # forward pass
    def forward(self, index, targets=None):

        # converts input sequence index into embeddings
        token_embeddings = self.token_embedding_table(index)
        # Position embeddings must match the batch size of token embeddings
        position_embeddings = self.position_embedding_table(torch.arange(index.size(1)))  # (sequence_length, num_embeddings)
        # Now expand position embeddings to match the batch size of token embeddings
        position_embeddings = position_embeddings.unsqueeze(0).expand(index.size(0), -1, -1)  # (batch_size, sequence_length, num_embeddings)
        # Combine token embeddings and position embeddings
        combined_embeddings = token_embeddings + position_embeddings  # Now both have shape (batch_size, sequence_length, num_embeddings)
        logits = self.linear_projection(self.head(combined_embeddings))

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
            logits, loss = self(index[:, -context_size:])
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1) # apply softmax to get probability distribution
            next_index = torch.multinomial(probs, num_samples=1) # retrieve sample from the distribution
            index = torch.cat((index, next_index), dim=1) # # append sample  to the full sequence
        return index