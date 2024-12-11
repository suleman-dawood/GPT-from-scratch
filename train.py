import torch
from constants import *
from model import *
from reader import *
from tokenizers import ByteLevelBPETokenizer
from nietzsche import all_text

# Load the tokenizer from files
tokenizer = ByteLevelBPETokenizer.from_file("vocab-vocab.json", "vocab-merges.txt")

# Function to encode the tokens
def encode(text):
    encoded = tokenizer.encode(text)
    return encoded.ids  # Return the token indices


# Debugging data preparation
torch.manual_seed(SEED)
data = torch.tensor(encode(all_text), dtype=torch.long)  # convert data into a tensor

# Split data into training and validation sets
split = int(DATA_SPLIT * len(data))
training_set = data[:split]  # split into training and validation set
validation_set = data[split:]
# Generate a small batch of data to process of inputs x and targets y
def get_batch(split):
    if split == 'train':
        data = training_set
    else:
        data = validation_set

    # generate a random starting point and generate a batch from that
    random_batch = torch.randint(len(data) - context_size, (batch_size,))
    x_list = []
    y_list = []

    for i in random_batch:
        x_list.append(data[i:i + context_size])
        y_list.append(data[i + 1:i + context_size + 1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x, y


# Evaluates loss for both validation and training sets for printing after a set period
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(val_iterations)
        for k in range(val_iterations):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model():
    sample_model = BigramModel()

    # Using AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(sample_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        # We compare training and validation loss every once in a while
        if epoch % val_iterations == 0:
            losses = estimate_loss(sample_model)
            print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Backpropagation step to compute gradients
        x_sample, y_sample = get_batch("train")
        logits, loss = sample_model(x_sample, y_sample)
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")  # Debugging: print loss after each batch

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(sample_model.parameters(), max_norm=GRAD_CLIP)

        optimizer.step()

    # Saving the trained model for use later on
    torch.save(sample_model.state_dict(), "trained_model.pth")

train_model()
