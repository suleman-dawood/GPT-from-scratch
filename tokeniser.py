import re
from collections import Counter
from nietzsche import *
from constants import *
from tokenizers import ByteLevelBPETokenizer

def cleanup(text):
    # Normalize and clean text
    text = re.sub(r"[‘’“”]", "'", text)  # Normalize quotes
    text = re.sub(r'([.,!?;:])', r" \1 ", text)  # Preserve punctuation
    text = re.sub(r"\b(\w+)'(\w+)\b", r"\1'\2", text)  # Preserve contractions
    text = re.sub(r'[-_—]', '', text)  # Remove dashes
    text = text.replace("\n", " <newline> ")  # Preserve newlines as tokens
    text = text.replace("\t", " <tab> ")  # Preserve tabs as tokens
    text = re.sub(r'([(){}=[\]])', r' \1 ', text)  # Add spaces around brackets
    text = re.sub(r'"', r' <quote> ', text)  # Preserve quotes
    text = text.replace("\ufeff", "")  # Remove BOM if present
    return text

# Preprocess and clean the text
all_text_cleaned = cleanup(all_text)

# Save the cleaned text to a file for tokenizer training
with open("cleaned_text.txt", "w", encoding="utf-8") as file:
    file.write(all_text_cleaned)

# Initialize the tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the cleaned text file
tokenizer.train(files=["cleaned_text.txt"], vocab_size=NUM_MERGES, min_frequency=2, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>", "<newline>", "<tab>", "<quote>"])

# Save the tokenizer model and vocabulary
tokenizer.save_model(".", "vocab")  # This will save the vocab files in the current directory

# tokenizer = ByteLevelBPETokenizer.from_file("vocab-vocab.json", "vocab-merges.txt")

''' Old tokenizer just didn;t work reliably
def get_frequencies(vocab):
    # Identify frequent pairs in the vocabulary
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    # Merge the most frequent pair into a single token
    replacement = ''.join(pair)
    new_vocab = {}
    bigram = ' '.join(pair)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]

    return new_vocab

def tokenization(text, num_merges):
    # Clean up the input text first
    text = cleanup(text)
    vocab = {}

    # Tokenize the text, ensuring <newline>, <tab>, and other special tokens are treated as standalone
    for word in text.split():
        word = ' '.join(list(word)) + ' </w>'
        vocab[word] = vocab.get(word, 0) + 1

    # Apply Byte Pair Encoding (BPE) for the specified number of merges
    for i in range(num_merges):
        pairs = get_frequencies(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)

    return vocab


# Apply BPE
vocab = tokenization(all_text, NUM_MERGES)

# Save vocabulary to file
with open("vocab.txt", "w", encoding="utf-8") as file:
    for token in vocab.keys():
        file.write(f"{token}\n")
'''