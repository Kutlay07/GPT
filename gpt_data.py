import kagglehub
import os
import tiktoken
import torch

from torch.utils.data import Dataset


def download_dataset():
    path = kagglehub.dataset_download(
        "vivekmettu/wikitext2-data"
    )
    print(f"Dataset has been downloaded to {path}")
    return path


def load_dataset(dataset_path):
    train_path = os.path.join(
        dataset_path,
        "train.txt"
    )

    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


enc = tiktoken.get_encoding("gpt2")


def clean_text(text):

    replacements = {
        "<unk>": "",
        "@-@": "-",
        "@.@": ".",
        "@,@": ","
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def tokenize(text):
    return enc.encode(text)


def decode(token_ids):
    return enc.decode(token_ids)


class GPTDataset(Dataset):

    def __init__(self, token_ids, block_size, stride):
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride


    def __len__(self):
        return (len(self.token_ids) - self.block_size) // self.stride


    def __getitem__(self, idx):

        start = idx * self.stride

        x = self.token_ids[
            start:start + self.block_size
        ]

        y = self.token_ids[
            start + 1:start + self.block_size + 1
        ]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y