from gpt_model import GPT
from config import *
from gpt_data import *

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# Save Directories
# ==========================================================
if "COLAB_GPU" in os.environ:
    from google.colab import drive

    drive.mount("/content/drive")

    ROOT_DIR = "/content/drive/MyDrive/GPT"

else:
    ROOT_DIR = "."

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ==========================================================
# Save Function
# ==========================================================
def verify_checkpoint(path):
    try:
        torch.load(path, map_location="cpu")
        print(f"✓ Verified: {os.path.basename(path)}")
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def save_checkpoint(path, epoch, optimizer, loss):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )

    verify_checkpoint(path)


def save_model(path):

    torch.save(model.state_dict(), path)

    verify_checkpoint(path)

# ==========================================================
# Model
# ==========================================================
model = GPT(
    vocab_size=VOCAB_SIZE,
    embed_size=EMBED_SIZE,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
).to(device)

# ==========================================================
# Dataset
# ==========================================================
dataset_path = download_dataset()
text = load_dataset(dataset_path)
text = clean_text(text)

token_ids = tokenize(text)

dataset = GPTDataset(
    token_ids,
    BLOCK_SIZE,
    stride = 64
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# ==========================================================
# Optimizer
# ==========================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

# ==========================================================
# Training
# ==========================================================
best_loss = float("inf")

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

        # ==========================================
        # Print Loss
        # ==========================================
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

        # ==========================================
        # Periodic Checkpoint
        # ==========================================
        if (batch_idx + 1) % 5000 == 0:

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"checkpoint_batch_{batch_idx+1}.pt",
            )

            print(f"\nSaving periodic checkpoint...")

            save_checkpoint(
                checkpoint_path,
                epoch + 1,
                optimizer,
                loss.item(),
            )

            print()

    avg_loss = epoch_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # ==========================================
    # Epoch Checkpoint
    # ==========================================
    epoch_checkpoint = os.path.join(
        CHECKPOINT_DIR,
        f"checkpoint_epoch_{epoch+1}.pt",
    )

    save_checkpoint(
        epoch_checkpoint,
        epoch + 1,
        optimizer,
        avg_loss,
    )

    # ==========================================
    # Best Model
    # ==========================================
    if avg_loss < best_loss:

        best_loss = avg_loss

        best_model_path = os.path.join(
            WEIGHTS_DIR,
            "best_model.pt",
        )

        save_model(best_model_path)

        print(f"New Best Model! Loss: {best_loss:.4f}")

    print(f"Best Loss: {best_loss:.4f}\n")

# ==========================================================
# Final Model
# ==========================================================
last_model_path = os.path.join(
    WEIGHTS_DIR,
    "last_model.pt",
)

save_model(last_model_path)

print("\nTraining Finished Successfully!")