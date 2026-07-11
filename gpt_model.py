import torch
from torch import nn

# Token Embedding
class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, embed_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size)
  
  def forward(self, tokens):
    # tokens -> (batch_size, block_size)
    x =  self.embedding(tokens)
    # x -> (batch_size, block_size, embed_size)
    return x