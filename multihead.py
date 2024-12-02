from head import *
import torch.nn as nn

# multiple heads running in parallel
class MultiHead(nn.Module):

    def __init__(self, head_count, head_size):
        super().__init__()
        self.heads = nn.ModuleList()  # initialize an empty ModuleList
        for _ in range(head_count):  # add each Head to the list
            self.heads.append(Head(head_size))

    def forward(self, x):

        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))  # apply each head to the input
        concatenated = torch.cat(head_outputs, dim=-1)  # concatenate the outputs
        return concatenated
