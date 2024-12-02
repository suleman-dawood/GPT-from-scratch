import torch.nn as nn
from torch.nn import functional
import torch
from constants import *
from multihead import *
from feedforward import *
from block import *
class BigramModel(nn.Module):

    def __init__(self):
        super().__init__()
        # converts token into dense vectors which capture semantic info/ context about the word
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, num_embeddings)
        # converts each position in the input sequence into dense vectore which captures info about sequence of tokens
        self.position_embedding_table = nn.Embedding(context_size, num_embeddings)
        self.blocks = nn.Sequential(
            Block(num_embeddings, head_count),
            Block(num_embeddings, head_count),
            Block(num_embeddings, head_count)
        )
        self.linear_projection = nn.Linear(num_embeddings, VOCAB_SIZE)
        self.loss_fn = nn.CrossEntropyLoss()

    # forward pass
    def forward(self, index, targets=None):

        # converts input sequence index into embeddings
        token_embeddings = self.token_embedding_table(index)
        # Position embeddings must match the batch size of token embeddings
        batch_size, seq_length = index.shape
        position_indices = torch.arange(seq_length, device=index.device)  # Match sequence length
        position_embeddings = self.position_embedding_table(position_indices).unsqueeze(0)  # Add batch dimension
        # Combine token embeddings and position embeddings
        combined_embeddings = token_embeddings + position_embeddings  # Now both have shape (batch_size, sequence_length, num_embeddings)
        logits = self.linear_projection(self.blocks(combined_embeddings)) # maps embeddings into a final output vector

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