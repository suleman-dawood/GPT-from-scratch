from constants import *
from reader import *
import torch

# character-to-index mapping
def create_mapping(tok):
    mapping = {}
    for i, token in enumerate(tok):
        mapping[token] = i
    return mapping

# reverse mapping for decoding
def create_reverse_mapping(encoder_map):
    reverse_map = {}
    for key, value in encoder_map.items():
        reverse_map[value] = key
    return reverse_map

tokens = read_tokens()  # Make sure this returns a list of tokens from your vocabulary

# Debugging: print the first 20 tokens to verify the token list
print("First 20 tokens:", tokens[:20])

encoder_map = create_mapping(tokens)
decoder_map = create_reverse_mapping(encoder_map)

# Debugging: print the first 20 token-to-index mappings
print("First 20 token-to-index mappings:")
for token, index in list(encoder_map.items())[:20]:
    print(f"Token: {token}, Index: {index}")

# Debugging: print the reverse mappings (index-to-token) for the first 20 indices
print("First 20 index-to-token mappings:")
for index, token in list(decoder_map.items())[:20]:
    print(f"Index: {index}, Token: {token}")

# Encode a string based on the encoder_map
def encode(string):
    mapping = encoder_map
    encoded = []
    print("Encoding string:", string[:20])  # Debugging input string

    # Tokenize the string before encoding (e.g., split by spaces or use BPE tokenizer)
    tokens = [word + "</w>" for word in string.split()] # Adjust this depending on your tokenizer setup (like BPE)

    # Debugging: Print the tokenized input
    print("Tokenized input:", tokens)

    for token in tokens:
        if token in mapping:
            encoded.append(mapping[token])
        else:
            print(f"Warning: Token '{token}' not found in encoder_map")  # Debugging missing token
            encoded.append(mapping.get('<unk>', -1))  # Use <unk> or -1 for unknown tokens

    print("Encoded output:", encoded[:20])  # Debugging encoded output
    return encoded

# Decode a list of indices using the reverse mapping
def decode(encoded):
    mapping = decoder_map
    decoded = []
    print("Decoding indices:", encoded)  # Debugging input indices

    for i in encoded:
        if i in mapping:
            word = mapping[i]
            word = word.replace("/<w>","")
            # Handle special tokens and replace with corresponding characters
            if word == "<newline>":
                decoded.append("\n")
            elif word == "<tab>":
                decoded.append("\t")
            elif word == "<quote>":
                decoded.append('"')
            else:
                decoded.append(word)
        else:
            print(f"Warning: Index '{i}' not found in decoder_map")  # Debugging missing index
            decoded.append('<unk>')  # Use <unk> for unknown indices

    # Join the decoded tokens into a string and replace </w> with space to handle word boundaries
    decoded_output = ''.join(decoded).replace('</w>', ' ')
    print("Decoded output:", decoded_output[:20])  # Debugging decoded output
    return decoded_output

# Sample text for testing
sample_text = "Save for his raucous, rhapsodical autobiography, Ecce Homo, "

# Debugging: print original text
print("Original text:", sample_text[:100])  # Print the first 100 characters

# Encoding the sample text
encoded_text = encode(sample_text)

# Decoding the encoded text
decoded_text = decode(encoded_text)

# Debugging: print the decoded text
print("Decoded text:", decoded_text[:100])  # Print the first 100 characters of the decoded text
