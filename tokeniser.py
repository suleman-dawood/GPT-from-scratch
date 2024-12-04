import re
from collections import Counter
from nietzsche import *
from constants import *

def cleanup(text):

    text = re.sub(r"[‘’“”]", "'", text)  # normalize quotes
    text = re.sub(r"([.,!?;])", r" \1 ", text)  # keep end of sentence, punctuations
    text = re.sub(r"\b(\w+)'(\w+)\b", r"\1'\2", text)  # preserve contractions (e.g. don't)
    text = re.sub(r'[-_—]', '', text) # remove dashes
    return text

def get_frequencies(vocab):
    # look for frequent pairs in the vocabulary
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    # merge the frequent pairs into singular tokens
    replacement = ''.join(pair)
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]

    return new_vocab

def tokenization(text, num_merges):
    # initial tokenization (split words into characters)
    text = cleanup(text)
    vocab = {}
    for word in text.split():
        word = ' '.join(list(word)) + ' </w>'  # add end-of-word token
        vocab[word] = vocab.get(word, 0) + 1

    # we develop a certain vocabulary size
    for i in range(num_merges):
        # we identify frequent pairs
        pairs = get_frequencies(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)

        # we merge the most frequent pair
        vocab = merge_vocab(best_pair, vocab)
        print(f"Merge {i + 1}/{num_merges}: Merged pair {best_pair}")

    return vocab


# Apply BPE
vocab = tokenization(all_text, NUM_MERGES)

with open("vocab.txt", "w", encoding="utf-8") as file:
    for token in vocab.keys():
        file.write(f"{token}\n")
