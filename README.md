# GPT From Scratch

A decoder-only GPT-style Transformer implemented entirely from scratch using PyTorch.

This project reproduces the core GPT architecture without relying on high-level libraries such as Hugging Face Transformers. Every major component, including token embeddings, causal self-attention, decoder blocks, training, and autoregressive text generation, is implemented manually for educational purposes.

---

# Architecture

![GPT Architecture](images/Architecture.png)

The model follows a decoder-only Transformer architecture similar to GPT.

Main components:

- Token Embeddings
- Learned Positional Embeddings
- Multi-Head Causal Self-Attention
- Feed Forward Network (MLP with GELU)
- Residual Connections
- Layer Normalization (Pre-LN)
- Language Modeling Head with Weight Tying

---

# Features

## Model Architecture

- Decoder-only Transformer (GPT-style)
- Multi-Head Causal Self-Attention
- Learned Token Embeddings
- Learned Positional Embeddings
- Feed Forward Network (MLP with GELU)
- Residual Connections
- Layer Normalization (Pre-LN)
- Weight Tying

## Data Pipeline

- WikiText-2 dataset support
- GPT-2 Byte Pair Encoding tokenizer (tiktoken)
- Dataset cleaning:
  - `<unk>`
  - `@-@`
  - `@.@`
  - `@,@`
- Sliding window dataset generation with configurable stride

## Training

- AdamW optimizer
- GPU training support (CUDA)
- Automatic checkpoint saving
- Best model tracking
- Google Drive checkpoint support for Colab training

## Text Generation

Supports:

- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (Nucleus) sampling

---

# Project Structure


.
├── config.py
├── gpt_data.py
├── gpt_model.py
├── train.py
├── generate.py
├── requirements.txt
├── README.md
├── outputs/
│ └── sample_generation.txt
└── images/
└── Architecture.png


---

# Model Configuration

| Parameter | Value |
|---|---|
| Vocabulary Size | 50,257 |
| Embedding Size | 768 |
| Number of Layers | 12 |
| Number of Heads | 12 |
| Context Length | 128 |
| Dropout | 0.1 |
| Optimizer | AdamW |

---

# Dataset

The model is trained on the WikiText-2 dataset.

Tokenization is performed using the GPT-2 Byte Pair Encoding tokenizer (tiktoken).

Training samples are generated using a sliding window approach with configurable stride.

---

# Training

Run training with:

```bash
python train.py

The training pipeline automatically:

Downloads the dataset
Cleans text artifacts
Tokenizes the data
Creates training sequences
Saves checkpoints and model weights
Text Generation

Generate text using:

python generate.py

Example:

output = generate(
    "Once upon a time",
    100,
    temperature=0.8,
    top_k=20,
    top_p=0.9
)

print(output)
Sample Output

Example generations produced by the trained model are available in:

outputs/sample_generation.txt
Version History
v1.1.0

Added:

Weight Tying
Temperature Sampling
Top-k Sampling
Top-p (Nucleus) Sampling
Dataset Cleaning
Sliding Window Dataset Generation
Improved Checkpoint Management
Google Drive Training Support
v1.0.0

Initial release:

GPT-style decoder-only Transformer
Causal self-attention
Training pipeline
Autoregressive generation
Future Improvements

Planned:

KV Cache
Rotary Positional Embeddings (RoPE)
Flash Attention
RMSNorm
SwiGLU
Mixed Precision Training (AMP)
Learning Rate Scheduler
Training Resume Support
Validation Loss Evaluation
Acknowledgements

This project is inspired by:

GPT-1 (OpenAI)
GPT-2 (OpenAI)
Attention Is All You Need
Andrej Karpathy's educational materials
