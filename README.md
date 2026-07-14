# GPT From Scratch

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

A decoder-only GPT-style Transformer implemented entirely from scratch using **PyTorch**.

This project reproduces the core GPT architecture without relying on high-level libraries such as Hugging Face Transformers. Every major component—including token embeddings, causal self-attention, decoder blocks, training, and autoregressive text generation—is implemented manually for educational purposes.

---

# 🏗️ Architecture

![GPT Architecture](images/Architecture.png)

The model follows the original GPT decoder-only Transformer architecture consisting of stacked decoder blocks with causal self-attention for autoregressive language modeling.

### Main Components

- Token Embeddings
- Learned Positional Embeddings
- Multi-Head Causal Self-Attention
- Feed Forward Network (MLP with GELU)
- Residual Connections
- Layer Normalization (Pre-LN)
- Language Modeling Head with Weight Tying

---

# ✨ Features

## Model Architecture

- Decoder-only Transformer (GPT-style)
- Multi-Head Causal Self-Attention
- Learned Token Embeddings
- Learned Positional Embeddings
- Feed Forward Network (MLP with GELU)
- Residual Connections
- Layer Normalization (Pre-LN)
- Weight Tying
- Automatic checkpoint verification

## Data Pipeline

- WikiText-2 dataset
- GPT-2 Byte Pair Encoding tokenizer (`tiktoken`)
- Dataset cleaning
  - `<unk>`
  - `@-@`
  - `@.@`
  - `@,@`
- Sliding window dataset generation with configurable stride

## Training

- AdamW optimizer
- GPU training (CUDA)
- Automatic checkpoint saving
- Best model tracking
- Google Drive checkpoint support for Google Colab

## Text Generation

Supports multiple decoding strategies:

- Greedy Decoding
- Temperature Sampling
- Top-k Sampling
- Top-p (Nucleus) Sampling

---

# 📁 Project Structure

```text
GPT-From-Scratch/
│
├── config.py
├── gpt_data.py
├── gpt_model.py
├── train.py
├── generate.py
├── requirements.txt
├── README.md
│
├── outputs/
│   └── sample_generation.txt
│
└── images/
    └── Architecture.png
```

---

# ⚙️ Model Configuration

| Parameter | Value |
|-----------|------:|
| Vocabulary Size | 50,257 |
| Embedding Size | 768 |
| Number of Layers | 12 |
| Number of Heads | 12 |
| Context Length | 128 |
| Dropout | 0.1 |
| Optimizer | AdamW |

---

# 📚 Dataset

The model is trained on the **WikiText-2** dataset.

Tokenization is performed using the GPT-2 Byte Pair Encoding tokenizer (`tiktoken`).

Training samples are created using a configurable sliding window strategy.

---

# 🚀 Training

Run training with:

```bash
python train.py
```

The training pipeline automatically:

- Downloads the dataset
- Cleans dataset artifacts
- Tokenizes the text
- Creates sliding-window training samples
- Saves checkpoints periodically
- Saves the best model weights
- Supports Google Drive checkpoints when training in Colab

---

# 💬 Text Generation

Run:

```bash
python generate.py
```

Example:

```python
output = generate(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_k=20,
    top_p=0.9,
)

print(output)
```

---

# 📄 Sample Output

Example generations are available in:

```text
outputs/sample_generation.txt
```

---

## 🚧 Current Progress

### Completed

- ✅ GPT-style decoder-only Transformer
- ✅ Multi-Head Causal Self-Attention
- ✅ GPT-2 tokenizer (tiktoken)
- ✅ Weight Tying
- ✅ Temperature Sampling
- ✅ Top-k Sampling
- ✅ Top-p (Nucleus) Sampling
- ✅ Dataset Cleaning
- ✅ Sliding Window Dataset
- ✅ Checkpoint Management
- ✅ Google Drive Support for Colab

### In Progress

- 🔄 Model Training & Evaluation

### Planned

- ⏳ Automatic Mixed Precision (AMP)
- ⏳ KV Cache
- ⏳ Learning Rate Scheduler
- ⏳ Resume Training
- ⏳ Validation Loss
- ⏳ Perplexity Evaluation
- ⏳ Rotary Positional Embeddings (RoPE)
- ⏳ Flash Attention
- ⏳ RMSNorm
- ⏳ SwiGLU

---

# 📝 Version History

## v1.1.0

### Added

- Weight Tying
- Temperature Sampling
- Top-k Sampling
- Top-p (Nucleus) Sampling
- Dataset Cleaning
- Sliding Window Dataset Generation
- Improved Checkpoint Management
- Google Drive Support for Colab Training

## v1.0.0

Initial release featuring:

- GPT-style decoder-only Transformer
- Multi-Head Causal Self-Attention
- Training Pipeline
- Autoregressive Text Generation

---

# 🔮 Future Improvements

Planned features:

- Automatic Mixed Precision (AMP)
- KV Cache
- Learning Rate Scheduler
- Resume Training
- Validation Loss
- Perplexity Evaluation
- Flash Attention
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SwiGLU

---

# 🙏 Acknowledgements

This project is inspired by:

- OpenAI GPT-1
- OpenAI GPT-2
- Attention Is All You Need
- Andrej Karpathy's educational materials