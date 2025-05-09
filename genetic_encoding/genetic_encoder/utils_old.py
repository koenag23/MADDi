import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random

class TokenizedVCFDataset(Dataset):
    def __init__(self,
                 csv_path,
                 subject_list=None,
                 format_fields=("GT", "GQ", "PLMODE"),
                 max_variants=None,
                 pad_token="[PAD]",
                 unk_token="[UNK]",
                 mask_token="[MASK]",
                 token_to_id=None):
        # --- load and identify subjects ---
        self.df = pd.read_csv(csv_path)
        fixed = ["#CHROM","ALT","FILTER","FORMAT","ID","INFO","POS","QUAL","REF"]
        all_subj = [c for c in self.df.columns if c not in fixed]
        self.subjects = subject_list if subject_list else all_subj

        # --- always subtokenize ---
        self.format_fields = format_fields

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        # --- max_variants = all variants × 2 subtokens/field ---
        default = len(self.df) * len(format_fields) * 2
        self.max_variants = max_variants if max_variants is not None else default

        # --- build or reuse vocab ---
        self.token_to_id = token_to_id or {
            pad_token: 0, unk_token: 1, mask_token: 2
        }
        self.id_to_token = list(self.token_to_id.keys())
        update_vocab = token_to_id is None

        # --- tokenize each subject, expand vocab if needed ---
        self.subject_sequences = {}
        for subj in self.subjects:
            toks = self._tokenize_subject(subj)
            self.subject_sequences[subj] = toks
            if update_vocab:
                for t in toks:
                    if t not in self.token_to_id:
                        self.token_to_id[t] = len(self.token_to_id)
                        self.id_to_token.append(t)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        toks = self.subject_sequences[subj]
        ids  = [self.token_to_id.get(t, self.token_to_id[self.unk_token]) for t in toks]

        # pad or truncate
        if len(ids) > self.max_variants:
            ids = ids[:self.max_variants]
        else:
            ids += [self.token_to_id[self.pad_token]] * (self.max_variants - len(ids))

        return torch.tensor(ids, dtype=torch.long), subj

    def _tokenize_subject(self, subj):
        tokens = []

        for i, cell in enumerate(self.df[subj].fillna("0/0:.:.:.:.")):
            snp_name = str(self.df.loc[i, "ID"])  # <- SNP identifier
            tokens.append(f"{snp_name}")

            parts = str(cell).split(":")

            try:
                # GT
                gt = parts[0]
                tokens += ["GT_", f"##{gt}"]

                # GQ
                if len(parts) >= 4:
                    gq = int(parts[3])
                    b = f"{(gq // 10) * 10}-{((gq // 10) + 1) * 10}"
                    tokens += ["GQ_", f"##{b}"]

                # PLMODE
                if len(parts) >= 5:
                    pl = [int(p) for p in parts[4].split(",") if p.strip().isdigit()]
                    if pl:
                        m = pl.index(min(pl))
                        tokens += ["PLMODE_", f"##{m}"]

            except Exception:
                tokens.append(self.unk_token)

            tokens.append("[SEP]")  # End of SNP block

        return tokens



class MinimalBERT(nn.Module):
    """
    BERT-style encoder with:
      - dropout on embeddings
      - adjustable feed-forward dropout
      - final LayerNorm on encoder output
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 768,
                 max_len: int = 1536,
                 num_layers: int = 12,
                 nhead: int = 12,
                 embed_dropout: float = 0.1,
                 ff_dropout: float = 0.2):
        super().__init__()
        # token + positional embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed   = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.embed_dropout = nn.Dropout(embed_dropout)

        # Transformer encoder layers with more FF dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=ff_dropout,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final normalization before the MLM head
        self.final_norm = nn.LayerNorm(embed_dim)

        # MLM head
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        x: (B, L) token IDs
        returns: (B, L, V) logits
        """
        emb = self.token_embed(x) + self.pos_embed[:, :x.size(1), :]
        emb = self.embed_dropout(emb)
        h   = emb.transpose(0, 1)                  # (L, B, D)
        out = self.encoder(h).transpose(0, 1)      # (B, L, D)
        out = self.final_norm(out)
        return self.output_head(out)


def mask_inputs(x, token_to_id, id_to_token, field_mask_probs):
    B, L = x.shape
    device = x.device

    # 1) Build a boolean mask of all “value” tokens (those starting with '##')
    maskable = torch.zeros_like(x, dtype=torch.bool)
    for idx, tok in enumerate(id_to_token):
        if tok.startswith("##"):
            maskable |= (x == idx)

    # 2) Build a vector `prob_map` so prob_map[field_id] == mask probability
    prob_map = torch.zeros(len(id_to_token), device=device)
    for field, p in field_mask_probs.items():
        prob_map[token_to_id[field]] = p

    # 3) Shifted mask: get per‑position mask probability from preceding token
    probs = prob_map[x[:, :-1]]         # shape (B, L-1)
    coin  = torch.rand((B, L-1), device=device)

    # 4) Decide which positions to mask
    to_mask = (coin < probs) & maskable[:, 1:]

    # 5) Prepare labels and masked inputs
    labels = torch.full_like(x, -100)
    labels[:, 1:][to_mask] = x[:, 1:][to_mask]

    x_masked = x.clone()
    b_idxs, pos_idxs = to_mask.nonzero(as_tuple=True)
    # 80% → [MASK]
    n = len(b_idxs)
    keep = torch.rand(n, device=device)
    mask80 = keep < 0.8
    x_masked[b_idxs[mask80], pos_idxs[mask80]+1] = token_to_id["[MASK]"]
    # 10% → random
    mask10 = (keep >= 0.8) & (keep < 0.9)
    rand_ids = torch.randint(0, len(id_to_token), (mask10.sum().item(),), device=device)
    x_masked[b_idxs[mask10], pos_idxs[mask10]+1] = rand_ids
    # 10% keep original

    return x_masked, labels
