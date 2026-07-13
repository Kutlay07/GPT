import torch

from config import *
from gpt_model import GPT
from gpt_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(
    vocab_size = VOCAB_SIZE,
    embed_size = EMBED_SIZE,
    block_size = BLOCK_SIZE,
    dropout = DROPOUT,
    num_heads = NUM_HEADS,
    num_layers = NUM_LAYERS,
).to(device)

model.load_state_dict(
    torch.load("weights/gpt_model.pt", map_location=device))
model.eval()
def generate(prompt, max_new_tokens):
    token_ids = tokenize(prompt)
    tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    tokens = tokens.to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(1) > BLOCK_SIZE:
                tokens = tokens[:, -BLOCK_SIZE:]
            logits = model(tokens)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=-1)

    token_ids = tokens.squeeze(0).tolist()
    text = decode(token_ids)
    return text

# Example of usage 
# if __name__ == "__main__":
#     output = generate("The", 100)
#     print(output)
