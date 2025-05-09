import torch
from torch import nn, Tensor
import bitsandbytes as bnb
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = bnb.nn.StableEmbedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.use_flash_attention = use_flash_attention
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        
        # Flash Attention Module (if available)
        if use_flash_attention:
            self.attn = flash_attn_varlen_qkvpacked_func
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        if self.use_flash_attention:
            # Flash Attention expects (B, S, E) format for queries, keys, values
            B, S, _ = x.shape
            
            # Convert attention_mask to padding mask (True = pad)
            if attention_mask is not None:
                key_padding_mask = attention_mask == 0
            else:
                key_padding_mask = None
            
            x_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(x, key_padding_mask)
            qkv = self.qkv_proj(x_unpad).reshape(-1, 3, self.num_heads, self.head_dim)

            if qkv.dtype != torch.float16:
                qkv = qkv.to(torch.float16)

            attn_output_unpad = self.attn(
                qkv, cu_seqlens, max_seqlen=max_seqlen, dropout_p=0.0, causal=False
            )
            attn_output = pad_input(attn_output_unpad, indices, B, S)
            x = attn_output.view(B, S, self.num_heads * self.head_dim)
        else:
            x, _ = self.attn(x, x, x, key_padding_mask=attention_mask)

        x = residual + self.dropout(x)

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x

class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
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

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        
        return self.ln_f(x)

class BERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.bert = Model(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_flash_attention=use_flash_attention
        )
        
        self.mlm_head   = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.mlm_head(x)
