from torch import nn, Tensor, arange
from flash_attn.modules.mha import MHA
import bitsandbytes as bnb

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = bnb.nn.StableEmbedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(seq_len, embed_dim)

    def forward(self, x):
        seq_len = x.size(1)  # (batch_size, seq_len, embedding_dim)
        positions = arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return self.position_embeddings(positions)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, use_flash_attention=True):
        super().__init__()
        self.use_flash_attention = use_flash_attention

        # Attention layer (FlashAttention or MHA)
        if use_flash_attention:
            self.attn = MHA(embed_dim, num_heads, dropout=dropout, causal=False)  # FlashAttention optimized MHA
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)

        # Optimized feedforward network
        self.ffn_layer1 = nn.Linear(embed_dim, dim_feedforward)
        self.ffn_layer2 = nn.Linear(dim_feedforward, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # Attention block (FlashAttention or MHA)
        if self.use_flash_attention:
            x = self.attn(x, key_padding_mask=attention_mask)
        else:
            residual = x
            x = self.norm1(x)  # Apply norm before attention (standard in MHA)
            attn_out, _ = self.attn(x, x, x, key_padding_mask=attention_mask)
            x = residual + attn_out

        # Feedforward block
        residual = x
        x = self.ffn_layer1(x)  # First linear layer
        x = self.gelu(x)  # GELU activation
        x = self.ffn_layer2(x)  # Second linear layer
        x = self.dropout(x)  # Dropout layer
        x = residual + x  # Add residual connection

        return x

class Bert(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        seq_len: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim,
                num_heads,
                dim_feedforward,
                dropout,
                use_flash_attention
            ) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.mlm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = self.embedding(input_ids)
        position_embeddings = self.positional_embedding(input_ids)
        
        x = x + position_embeddings
        
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        return x, self.mlm_head(x)
