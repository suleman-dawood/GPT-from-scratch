from model import BigramModel
import torch
from constants import *
from tokenizers import ByteLevelBPETokenizer

def generate_text(out_length):
    # Initialize the model and load the pre-trained weights
    sample_model = BigramModel()
    sample_model.load_state_dict(torch.load("trained_model.pth"))
    sample_model.eval()

    # Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer.from_file("vocab-vocab.json", "vocab-merges.txt")

    # Generating sample
    initial = torch.zeros((1, context_size), dtype=torch.long)
    sample_generation = sample_model.generate(initial, new_tokens=out_length)[0].tolist()

    # Decoding the generated token indices
    decoded_text = tokenizer.decode(sample_generation)
    lines = decoded_text.split('\n')
    if len(lines) > 1:
        decoded_text = '\n'.join(lines[1:]).strip()

    # Write the generated text to a file
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(decoded_text)

generate_text(120)
