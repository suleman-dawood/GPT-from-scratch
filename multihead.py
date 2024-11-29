from head import *
import torch.nn as nn

# multiple heads running in parallel
class MultiHead(nn.Module):

    def __init__(self, head_count, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_count)])

    def forward(self, x):
        concatenated = torch.cat([head(x) for head in self.heads], dim=-1)
        return concatenated

