from gpt_model import GPT
from config import *
from gpt_data import *

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model
# =========================
model = GPT(
    vocab_size=VOCAB_SIZE,
    embed_size=EMBED_SIZE,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(device)

# =========================
# Dataset
# =========================
dataset_path = download_dataset()
text = load_dataset(dataset_path)
token_ids = tokenize(text)

dataset = GPTDataset(token_ids, BLOCK_SIZE)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# Optimizer
# =========================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================
# Training
# =========================
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        B, T, V = logits.shape

        logits = logits.reshape(B * T, V)
        targets = y.reshape(B * T)

        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = epoch_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, f"checkpoint_epoch_{epoch+1}.pt")

    print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pt\n")
    
torch.save(model.state_dict(), "gpt_final.pt")
print("Final model saved.")