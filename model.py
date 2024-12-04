import torch.nn as nn
import torch
from constants import *
from multihead import *
from feedforward import *
from block import *
class BigramModel(nn.Module):

    def __init__(self):
        super().__init__()
        # converts token into dense vectors which capture semantic info context about the word
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, num_embeddings)
        # converts each position in the input sequence into dense vectors which captures info about sequence of tokens
        self.position_embedding_table = nn.Embedding(context_size, num_embeddings)
        blocks = []
        for i in range(layer_count):
            blocks.append(Block(num_embeddings, head_count))

        # pass the list to nn.Sequential and unpack
        self.blocks = nn.Sequential(*blocks)
        self.ln1 = nn.LayerNorm(num_embeddings) # It normalizes the activations for a given layer along the last dimension (num_embeddings)
        self.linear_projection = nn.Linear(num_embeddings, VOCAB_SIZE) #  converts the normalized embeddings into logits that can be interpreted as probabilities
        self.apply(self._init_weights)
        self.loss_fn = nn.CrossEntropyLoss()

    # used to initialize the weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): # initialise linear layers using a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # initialise Embedding layer using a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm): # initialise LayerNorm to weight of 1 and bias of 0 to get a good start
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    # forward pass
    def forward(self, index, targets=None):

        # converts input sequence index into embeddings
        token_embeddings = self.token_embedding_table(index)
        # Position embeddings must match the batch size of token embeddings
        batch_length, seq_length = index.shape
        position_indices = torch.arange(seq_length)  # Match sequence length
        position_embeddings = self.position_embedding_table(position_indices).unsqueeze(0)  # Add batch dimension
        # Combine token embeddings and position embeddings
        combined_embeddings = token_embeddings + position_embeddings  # Now both have shape (batch_size, sequence_length, num_embeddings)
        logits = self.linear_projection(self.blocks(self.ln1(combined_embeddings))) # maps embeddings into a final output vector

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

        for i in range(new_tokens): # predicts next token based on current sequence new_tokens amount of times
            logits, loss = self(index[:, -context_size:]) # trucate to include only relevant context
            logits = logits[:, -1, :] # predictions for the last token
            probs = nn.functional.softmax(logits, dim=-1) # apply softmax to get probability distribution
            next_index = torch.multinomial(probs, num_samples=1) # retrieve sample from the distribution
            index = torch.cat((index, next_index), dim=1) # # append sample  to the full sequence
        return index