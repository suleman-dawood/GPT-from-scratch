import torch
from mappings import *
from constants import *
from model import *


encoder_map = create_mapping()
decoder_map = create_reverse_mapping(encoder_map)


torch.manual_seed(SEED)
data = torch.tensor(encode(full_text, encoder_map), dtype=torch.long) # convert data into a tensor

split = int(DATA_SPLIT*len(data))
training_set = data[:split] # split into training and validation set
validation_set = data[split:]

# generate a small batch of data to process of inputs x and targets y
def get_batch(split):

    if split == 'train':
        data = training_set
    else:
        data =  validation_set

    # generate a random starting point and generate a batch from that
    random_batch = torch.randint(len(data) - context_size, (batch_size,))
    x_list = []
    y_list = []

    for i in random_batch:
        x_list.append(data[i:i+context_size])
        y_list.append(data[i+1:i+context_size+1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x, y

x_sample, y_sample = get_batch('train')

# evaluates loss for both validation and trainiing sets
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

sample_model = BigramModel()

optimizer = torch.optim.Adam(sample_model.parameters(), lr = LR)

for epoch in range(EPOCHS):

    # we compare training and validation loss every once in a while
    if epoch % val_iterations == 0:
        losses = estimate_loss(sample_model)
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    x_sample, y_sample = get_batch("train")

    # backpropogation step to compute gradients
    logits, loss = sample_model(x_sample, y_sample)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

initial = torch.zeros((1, context_size), dtype=torch.long)
sample_generation = sample_model.generate(initial, new_tokens=NEW_TOKENS)[0].tolist()
print(decode(sample_generation, decoder_map))


