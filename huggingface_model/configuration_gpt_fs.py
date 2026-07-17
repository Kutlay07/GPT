from transformers import PretrainedConfig


class GPTFromScratchConfig(PretrainedConfig):
    model_type = "gpt_from_scratch"
    tie_word_embeddings = True

    def __init__(
        self,
        vocab_size=50257,
        embed_size=768,
        block_size=128,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = embed_size
        self.block_size = block_size
        self.num_attention_heads = num_heads
        self.num_hidden_layers = num_layers
        self.dropout = dropout
        self.max_position_embeddings = block_size
        self.tie_word_embeddings = True