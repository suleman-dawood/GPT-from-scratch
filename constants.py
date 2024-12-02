# reading file
with open("input.txt", 'r', encoding='utf-8') as file:
    full_text = file.read()


characters = sorted(set(full_text))
VOCAB_SIZE = len(characters)
DATA_SPLIT = 0.9
NEW_TOKENS = 500
SEED = 1
LR = 0.0005
EPOCHS = 500
DROPOUT = 0.2
context_size = 256
batch_size = 64
val_iterations = 10
num_embeddings = 384
head_count = 6
layer_count = 6