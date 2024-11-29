# reading file
with open("input.txt", 'r', encoding='utf-8') as file:
    full_text = file.read()


characters = sorted(set(full_text))
VOCAB_SIZE = len(characters)
DATA_SPLIT = 0.8
SEED = 1
LR = 0.001
EPOCHS = 1000
DROPOUT = 0.2
context_size = 16
batch_size = 32
val_iterations = 500
num_embeddings = 32