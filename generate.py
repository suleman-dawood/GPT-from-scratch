from model import BigramModel
import torch
from constants import *
from mappings import *

def generate_text():
    # initializing the model and load the pre-trained weights
    sample_model = BigramModel()
    sample_model.load_state_dict(torch.load("trained_model.pth"))
    sample_model.eval()

    # generating sample
    initial = torch.zeros((1, context_size), dtype=torch.long)
    sample_generation = sample_model.generate(initial, new_tokens=NEW_TOKENS)[0].tolist()

    # decoding the generated text
    encoder_map = create_mapping()
    decoder_map = create_reverse_mapping(encoder_map)
    print(decode(sample_generation, decoder_map))


generate_text()