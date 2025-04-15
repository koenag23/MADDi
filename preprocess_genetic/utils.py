import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TokenizedVCFDataset(Dataset):
    def __init__(self, csv_path, subject_list=None, format_fields=("GT", "GQ", "PLMODE"),
                 max_variants=None, pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]",
                 token_to_id=None, use_subtoken_strategy=False):

        self.df = pd.read_csv(csv_path)
        self.fixed_cols = ["#CHROM", "ALT", "FILTER", "FORMAT", "ID", "INFO", "POS", "QUAL", "REF"]

        all_subjects = [col for col in self.df.columns if col not in self.fixed_cols]
        self.subjects = subject_list if subject_list else all_subjects
        self.format_fields = format_fields
        self.use_subtoken_strategy = use_subtoken_strategy

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        # Handle dynamic max_variants
        default_variant_count = len(self.df) * (2 if use_subtoken_strategy else 1) * len(format_fields)
        self.max_variants = max_variants if max_variants is not None else default_variant_count

        # Vocabulary setup
        self.token_to_id = token_to_id or {pad_token: 0, unk_token: 1, mask_token: 2}
        self.id_to_token = list(self.token_to_id.keys())
        update_vocab = token_to_id is None

        # Precompute subject token sequences
        self.subject_sequences = {}
        for subj in self.subjects:
            tokens = self._tokenize_subject(subj)
            self.subject_sequences[subj] = tokens
            if update_vocab:
                for tok in tokens:
                    if tok not in self.token_to_id:
                        self.token_to_id[tok] = len(self.token_to_id)
                        self.id_to_token.append(tok)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        tokens = self.subject_sequences[subj]
        token_ids = [self.token_to_id.get(tok, self.token_to_id[self.unk_token]) for tok in tokens]

        if len(token_ids) > self.max_variants:
            token_ids = token_ids[:self.max_variants]
        else:
            token_ids += [self.token_to_id[self.pad_token]] * (self.max_variants - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long), subj

    def _tokenize_subject(self, subj):
        tokens = []
        for cell in self.df[subj].fillna("0/0:.:.:.:."):
            raw = str(cell).split(":")
            try:
                if "GT" in self.format_fields:
                    gt_val = raw[0]
                    tokens.extend(["GT_", f"##{gt_val}"] if self.use_subtoken_strategy else [f"GT_{gt_val}"])

                if "GQ" in self.format_fields and len(raw) >= 4:
                    gq = int(raw[3])
                    gq_bin = f"{(gq // 10) * 10}-{((gq // 10) + 1) * 10}"
                    tokens.extend(["GQ_", f"##{gq_bin}"] if self.use_subtoken_strategy else [f"GQ_{gq_bin}"])

                if "PLMODE" in self.format_fields and len(raw) >= 5:
                    pl = [int(p) for p in raw[4].split(",") if p.strip().isdigit()]
                    if pl:
                        pl_mode = pl.index(min(pl))
                        tokens.extend(["PLMODE_", f"##{pl_mode}"] if self.use_subtoken_strategy else [f"PLMODE_{pl_mode}"])
            except Exception:
                tokens.append(self.unk_token)

        return tokens

class MinimalBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, max_len=1536, num_layers=6, nhead=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward=256, dropout=0.1),
            num_layers=num_layers
        )
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        Args:
          x: (B, L) token ids
        Returns:
          logits: (B, L, vocab_size)
        """
        emb = self.token_embed(x) + self.pos_embed[:, :x.size(1), :]
        x = emb.transpose(0, 1)  # (L, B, D)
        out = self.encoder(x).transpose(0, 1)  # (B, L, D)
        logits = self.output_head(out)
        return logits

def mask_inputs(x, vocab_size, mask_token_id, pad_token_id, mask_prob=0.15):
    """
    Masks input tokens in the batch for BERT-style pretraining.

    Args:
        x: Tensor of shape (B, L) containing token ids.
        vocab_size: Size of the vocabulary.
        mask_token_id: Token id used for masking.
        pad_token_id: Token id used for padding.
        mask_prob: The probability to mask a token.

    Returns:
        A tuple (masked_inputs, labels) where:
          - masked_inputs: Tensor with masked tokens.
          - labels: Tensor with original token ids for masked positions, else -100.
    """
    x_masked = x.clone()
    labels = torch.full_like(x, -100)
    
    # Generate a mask for tokens (non-padding) that will be selected for masking
    prob_matrix = torch.rand(x.shape, device=x.device)
    mask = (prob_matrix < mask_prob) & (x != pad_token_id)
    
    # Set labels for positions that will be masked
    labels[mask] = x[mask]
    
    # For masked tokens: 80% -> [MASK] token, 10% -> random token, 10% -> keep original
    rand_prob = torch.rand(x.shape, device=x.device)
    
    # 80% replacement with mask token
    mask_token_mask = mask & (rand_prob < 0.8)
    x_masked[mask_token_mask] = mask_token_id
    
    # 10% replacement with random token id
    random_token_mask = mask & (rand_prob >= 0.8) & (rand_prob < 0.9)
    random_tokens = torch.randint(0, vocab_size, x[random_token_mask].shape, device=x.device)
    x_masked[random_token_mask] = random_tokens
    
    # 10% chance: tokens remain unchanged (no action needed)

    return x_masked, labels