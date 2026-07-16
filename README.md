# GPT From Scratch

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

A decoder-only GPT-style Transformer implemented entirely from scratch using **PyTorch**.

The goal of this project is to understand every major component behind modern Large Language Models by implementing them manually without relying on high-level libraries such as Hugging Face Transformers.

The implementation now includes several modern architectural improvements such as **Rotary Position Embeddings (RoPE)**, **Flash Attention**, **Mixed Precision Training**, and **Weight Tying**, making it significantly closer to modern GPT architectures.

---

# 🏗️ Architecture

![GPT Architecture](images/Architecture.png)

The model follows the decoder-only Transformer architecture introduced by OpenAI GPT.

### Main Components

- Token Embeddings
- Rotary Position Embeddings (RoPE)
- Multi-Head Causal Self-Attention
- Flash Attention (PyTorch SDPA)
- Feed Forward Network (MLP with GELU)
- Residual Connections
- Layer Normalization (Pre-LN)
- Weight Tying
- Language Modeling Head

## 📊 Feature Status

| Feature | Status |
|---------|:------:|
| Decoder-only GPT | ✅ |
| Multi-Head Causal Self-Attention | ✅ |
| Rotary Position Embeddings (RoPE) | ✅ |
| Flash Attention (PyTorch SDPA) | ✅ |
| MLP with GELU | ✅ |
| Pre-LayerNorm | ✅ |
| Weight Tying | ✅ |
| GPT-2 Tokenizer (`tiktoken`) | ✅ |
| Automatic Mixed Precision (AMP) | ✅ |
| Gradient Clipping | ✅ |
| Validation Loss | ✅ |
| Perplexity Evaluation | ✅ |
| Resume Training | ✅ |
| Sliding Window Dataset | ✅ |
| Top-k Sampling | ✅ |
| Top-p Sampling | ✅ |
| Temperature Sampling | ✅ |
| Learning Rate Scheduler | ⏳ |
| KV Cache | ⏳ |
| Hugging Face Export | ⏳ |
| RMSNorm | ⏳ |
| SwiGLU | ⏳ |
---

# ✨ Features

## Model Architecture

- Decoder-only GPT architecture
- Multi-Head Causal Self-Attention
- Rotary Position Embeddings (RoPE)
- Flash Attention
- Feed Forward Network (MLP + GELU)
- Residual Connections
- Pre-LayerNorm
- Weight Tying

---

## Data Pipeline

- WikiText-2 dataset
- GPT-2 tokenizer (`tiktoken`)
- Dataset cleaning
    - `<unk>`
    - `@-@`
    - `@.@`
    - `@,@`
- Sliding Window dataset generation
- Configurable stride

---

## Training

- AdamW optimizer
- Automatic Mixed Precision (AMP)
- Gradient Clipping
- Validation Split
- Validation Loss
- Perplexity Evaluation
- Resume Training
- torch.compile() acceleration
- GPU Training (CUDA)
- Automatic checkpoint saving
- Best model tracking

---

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
├── checkpoints/
├── weights/
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

Tokenization is performed using the GPT-2 Byte Pair Encoding tokenizer provided by **tiktoken**.

Training samples are generated using a configurable sliding window strategy with adjustable stride.

---

# 🚀 Training

Run training with

```bash
python train.py
```

The training pipeline automatically

- Downloads the dataset
- Cleans dataset artifacts
- Tokenizes the text
- Creates sliding-window training samples
- Splits the dataset into training and validation sets
- Computes Validation Loss and Perplexity
- Saves checkpoints
- Tracks the best model
- Supports checkpoint resume
---

# 💬 Text Generation

Generate text with

```bash
python generate.py
```

Example

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

The generator supports multiple decoding strategies including Greedy Decoding, Temperature Sampling, Top-k Sampling, and Top-p (Nucleus) Sampling.

---

# 📄 Sample Output

Example generations can be found in

```text
outputs/sample_generation.txt
```

---

# 🚧 Current Progress

## ✅ Completed

### Model Architecture

- Decoder-only GPT Transformer
- Multi-Head Causal Self-Attention
- Rotary Position Embeddings (RoPE)
- Flash Attention (PyTorch SDPA)
- Feed Forward Network (MLP with GELU)
- Residual Connections
- Pre-LayerNorm
- Weight Tying

### Training Pipeline

- AdamW Optimizer
- Automatic Mixed Precision (AMP)
- Gradient Clipping
- Validation Split
- Validation Loss
- Perplexity Evaluation
- Resume Training
- Automatic Checkpoint Saving
- Best Model Tracking
- torch.compile() Support

### Data Pipeline

- WikiText-2 Dataset
- GPT-2 Tokenizer (tiktoken)
- Dataset Cleaning
- Sliding Window Dataset Generation
- Configurable Stride

### Text Generation

- Greedy Decoding
- Temperature Sampling
- Top-k Sampling
- Top-p (Nucleus) Sampling

---

## 🔄 In Progress

- Model Training & Evaluation

---

## 📌 Planned

- Learning Rate Scheduler
- KV Cache
- Hugging Face Model Export
- RMSNorm
- SwiGLU

---

# 📝 Version History

## v1.2.0

### Added

- Rotary Position Embeddings (RoPE)
- Flash Attention
- Automatic Mixed Precision (AMP)
- Gradient Clipping
- Validation Split
- Validation Loss
- Perplexity Evaluation
- Resume Training
- torch.compile() acceleration

### Improved

- Modernized GPT architecture
- Faster attention computation
- Faster training
- Improved checkpoint compatibility
- Improved evaluation pipeline

---

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

---

## v1.0.0

Initial implementation featuring

- GPT-style Decoder-only Transformer
- Multi-Head Causal Self-Attention
- Training Pipeline
- Autoregressive Text Generation

---

# 🔮 Future Improvements

Planned features include

- Learning Rate Scheduler
- KV Cache
- Hugging Face Model Export
- RMSNorm
- SwiGLU

---

# 🙏 Acknowledgements

This project is inspired by

- OpenAI GPT-1
- OpenAI GPT-2
- Attention Is All You Need
- RoFormer: Enhanced Transformer with Rotary Position Embedding
- FlashAttention
- Andrej Karpathy's educational materials