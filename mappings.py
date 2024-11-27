from constants import *
# character-to-index mapping
def create_mapping():
    mapping = {}
    for i, ch in enumerate(characters):
        mapping[ch] = i
    return mapping

# encode a string using the mapping
def encode(string, mapping):
    encoded = []
    for c in string:
        encoded.append(mapping[c])
    return encoded

# reverse mapping for decoding
def create_reverse_mapping(mapping):
    reverse_mapping = {}
    for ch, i in mapping.items():
        reverse_mapping[i] = ch
    return reverse_mapping

# decode an encoded list back to the original string
def decode(encoded, reverse_mapping):
    decoded = []
    for i in encoded:
        decoded.append(reverse_mapping[i])
    return ''.join(decoded)