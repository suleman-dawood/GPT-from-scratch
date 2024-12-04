from head import *
import torch.nn as nn

# multiple heads running in parallel
class MultiHead(nn.Module):

    def __init__(self, head_count, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_count)]) # each head independatly computes attention
        self.proj = nn.Linear(head_size * head_count, num_embeddings) # projects in into a vector embedding
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))  # apply each head to the input
        concatenated = torch.cat(head_outputs, dim=-1)  # concatenate the outputs
        return self.dropout(self.proj(concatenated))
