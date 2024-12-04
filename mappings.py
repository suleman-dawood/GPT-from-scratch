from constants import *
# character-to-index mapping
def create_mapping(tokens):
    mapping = {}
    for i, token in enumerate(tokens):
        mapping[token] = i
    return mapping


# encode a string using the mapping
def encode(string, mapping):
    encoded = []
    for c in string:
        encoded.append(mapping[c])
    return encoded

# reverse mapping for decoding
def create_reverse_mapping(tokens):
    mapping = {}
    for i, token in enumerate(tokens):
        mapping[i] = token
    return mapping

# decode an encoded list back to the original string
def decode(encoded, mapping):
    decoded = []
    for i in encoded:
        decoded.append(mapping[i])
    return ''.join(decoded).replace('</w>', ' ')