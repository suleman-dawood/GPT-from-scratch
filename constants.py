from reader import *

VOCAB_SIZE = len(read_tokens())
DATA_SPLIT = 0.9
NUM_MERGES = 30000
SEED = 6
LR = 0.0005
EPOCHS = 500
DROPOUT = 0.2
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
context_size = 128
batch_size = 32
val_iterations = 50
num_embeddings = 96
head_count = 4
layer_count = 4
