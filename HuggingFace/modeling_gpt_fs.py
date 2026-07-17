import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_gpt_fs import GPTFromScratchConfig

MAX_SEQ_LEN = 4096

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, tokens):
        # tokens -> (batch_size, block_size)
        x =  self.embedding(tokens)
        # x -> (batch_size, block_size, embed_size)
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_size))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # we normally write 1 / torch.sqrt(x) but torch.rsqrt(x) does it more efficiently
        return x * rms * self.weight

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x = torch.stack((-x2,x1),dim=-1)

    return x.flatten(-2)

def precompute_freqs(head_size, block_size):
    theta = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))

    positions = torch.arange(block_size).float()

    freqs = torch.outer(positions, theta)
    freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return freqs

class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        assert embed_size % num_heads == 0 # embed_size must be divisible by num_heads
        self.head_size = embed_size // num_heads
        self.c_attn = nn.Linear(embed_size, 3 * embed_size)
        self.c_proj = nn.Linear(embed_size, embed_size)

        freqs = precompute_freqs(self.head_size, MAX_SEQ_LEN)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)


    def forward(self, x, past_k=None, past_v=None):
        B, T, C = x.shape
        # x: (B, T, C)

        q, k, v = self.c_attn(x).split(self.embed_size, dim=2)
        # c_attn -> (B,T,3C) -> split -> Q(B,T,C) K(B,T,C) V(B,T,C)

        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        # (B, T, 768) -> (B,T, 12, 64) -> (B, 12, T, 64)


        past_len = 0 if past_k is None else past_k.size(2)
        
        cos = self.cos_cached[past_len : past_len + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[past_len : past_len + T].unsqueeze(0).unsqueeze(0)
        # cos,sin: (1, 1, T, D)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        # q,k: (B, H, T, D)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            
        if past_v is not None:
            v = torch.cat([past_v, v], dim=2)
            
        if past_k is not None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        # Q(B, H, T, D), K(B, H, T, D), V(B, H, T, D)
        # -> Flash Attention -> (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, C) 
        # (B,H,T,D) -> (B,T,H,D) -> (B,T,C)

        out = self.c_proj(out)
        # (B,T,C) -> c_proj -> (B,T,C)

        out = self.resid_dropout(out)
        # (B,T,C) -> (B,T,C)
        
        new_k = k # (B, H, T_total, D)
        new_v = v # (B, H, T_total, D)
        
        return out, new_k, new_v

class MLP(nn.Module):
    def __init__(self, embed_size,dropout):
        super().__init__()
        hidden_size = int((8 * embed_size) / 3)
        hidden_size = ((hidden_size + 63) // 64) * 64 # to make it hardware-friendly
        self.gate_proj = nn.Linear(embed_size, hidden_size, bias=False) # creates the gate
        self.up_proj = nn.Linear(embed_size, hidden_size, bias=False) # creates the information
        self.down_proj = nn.Linear(hidden_size, embed_size, bias=False) # reduces it back to the embedding dimension
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        up = self.up_proj(x) # (B,T,C)
        gate = self.gate_proj(x) # (B,T,embed_size) -> (B,T,hidden_size)
        gate = self.silu(gate) # -> (B,T,hidden_size) -> (B,T,hidden_size)
        x = gate * up # Element-wise multiplication
        x = self.down_proj(x) # (B,T,hidden_size) -> (B,T,embed_size)
        x = self.dropout(x) # (B,T,embed_size) -> (B,T,embed_size)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        self.rms_norm1 = RMSNorm(embed_size)
        self.attn = MultiHeadCausalSelfAttention(embed_size, num_heads, block_size, dropout)
        self.rms_norm2 = RMSNorm(embed_size)
        self.mlp = MLP(embed_size, dropout)

    def forward(self, x, past_k=None, past_v=None):
        
        attn, new_k, new_v = self.attn(
        self.rms_norm1(x),
        past_k=past_k,
        past_v=past_v,
        )
        
        x = x + attn
        
        x = x + self.mlp(self.rms_norm2(x))
        return x, new_k, new_v
    
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, dropout, num_heads, num_layers):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(embed_size, num_heads, block_size, dropout) 
            for _ in range(num_layers)])
        self.rms_norm_f = RMSNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False) # (B,T,C) -> (B,T,V(vocab_size))
        # Weight Tying
        self.lm_head.weight = self.token_embedding.embedding.weight # (V,C)

    def forward(self, tokens, past_kvs=None, use_cache=False,):
        if use_cache:
            if past_kvs is None:
                past_kvs = [(None, None) for _ in range(len(self.blocks))]

        new_past_kvs = []
    
        # tokens -> (B, T)
        x = self.token_embedding(tokens) # -> (B, T, C)
        x = self.dropout(x)

        if use_cache:
            for block, (past_k, past_v) in zip(self.blocks, past_kvs):
                x, new_k, new_v = block(
                x,
                past_k=past_k,
                past_v=past_v,
                )
                new_past_kvs.append((new_k, new_v))
                
            else:
                for block in self.blocks:
                    x, _, _ = block(x)

            x = self.rms_norm_f(x)

            logits = self.lm_head(x) # (B,T,C) -> (B,T,V(vocab_size))
            
            if use_cache:
                return logits, new_past_kvs
                
            return logits

class GPTFromScratchForCausalLM(PreTrainedModel):
    config_class = GPTFromScratchConfig
    base_model_prefix = "transformer"

    _tied_weights_keys = {
        "transformer.lm_head.weight": "transformer.token_embedding.embedding.weight",
    }
    
    def post_init(self):
        super().post_init()
        self.tie_weights()
        
    def __init__(self, config):
        super().__init__(config)

        self.transformer = GPT(
            vocab_size=config.vocab_size,
            embed_size=config.hidden_size,
            block_size=config.block_size,
            dropout=config.dropout,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.token_embedding.embedding

    def set_input_embeddings(self, value):
        self.transformer.token_embedding.embedding = value

    def get_output_embeddings(self):
        return self.transformer.lm_head
    
    def tie_weights(self, **kwargs):
        self.transformer.lm_head.weight = (
            self.transformer.token_embedding.embedding.weight
        )

        self._tied_weights_keys = {
            "transformer.lm_head.weight": "transformer.token_embedding.embedding.weight"
        }
    def set_output_embeddings(self, new_embeddings):
        self.transformer.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        if return_dict is None:
            return_dict = self.config.use_return_dict

        if use_cache is None:
            use_cache = False

        if use_cache:
            logits, new_past_key_values = self.transformer(
                input_ids,
                past_kvs=past_key_values,
                use_cache=True,
            )
        else:
            logits = self.transformer(
                input_ids,
                use_cache=False,
            )
            new_past_key_values = None

        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:

            output = (
                logits,
                new_past_key_values,
            )

            if loss is not None:
                return (loss,) + output

            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Beam search sırasında Hugging Face tarafından çağrılır.
        """
        if past_key_values is None:
            return None

        reordered_past = []

        for layer_past in past_key_values:
            k, v = layer_past

            reordered_past.append(
                (
                    k.index_select(0, beam_idx),
                    v.index_select(0, beam_idx),
                )
            )

        return reordered_past

    @property
    def dummy_inputs(self):
        device = self.device

        return {
            "input_ids": torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=device,
            )
        }

    def can_generate(self):
        return True