from nietzsche import *
from reader import *

full_text = all_text
VOCAB_SIZE = len(read_tokens())
DATA_SPLIT = 0.9
NEW_TOKENS = 1000
NUM_MERGES = 25000
SEED = 3
LR = 0.0005
EPOCHS = 1000
DROPOUT = 0.2
context_size = 128
batch_size = 32
val_iterations = 50
num_embeddings = 192
head_count = 3
layer_count = 3
