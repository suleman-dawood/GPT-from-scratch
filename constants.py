from nietzsche import *

full_text = all_text
characters = sorted(set(full_text))
VOCAB_SIZE = len(characters)
DATA_SPLIT = 0.9
NEW_TOKENS = 1000
SEED = 3
LR = 0.0005
EPOCHS = 5000
DROPOUT = 0.2
context_size = 256
batch_size = 32
val_iterations = 50
num_embeddings = 384
head_count = 3
layer_count = 3