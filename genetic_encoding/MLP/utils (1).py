import pandas as pd
import os
import math
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BigBirdConfig, BigBirdForMaskedLM
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import pandas as pd

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
        self.max_variants = max_variants

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
            #snp_name = str(self.df.loc[i, "ID"])  # <- SNP identifier
            #tokens.append(f"{snp_name}")

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

class ChunkedVCFDataset(Dataset):
    def __init__(self, base_ds, max_len):
        self.pad_id   = base_ds.token_to_id[base_ds.pad_token]
        self.unk_id   = base_ds.token_to_id[base_ds.unk_token]
        self.segments = []  # List of (ptid, [token_ids…])
        for ptid in base_ds.subjects:
            toks = base_ds.subject_sequences[ptid]
            ids  = [base_ds.token_to_id.get(t, self.unk_id) for t in toks]
            for i in range(0, len(ids), max_len):
                chunk = ids[i : i + max_len]
                if len(chunk) < max_len:
                    chunk += [self.pad_id] * (max_len - len(chunk))
                self.segments.append((ptid, chunk))

    def __len__(self): 
        return len(self.segments)

    def __getitem__(self, idx):
        ptid, chunk = self.segments[idx]
        return torch.tensor(chunk, dtype=torch.long), ptid

# 3) Simple 2‐layer head for classification
class PatientClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes=3, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.dropout  = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size//2, num_classes)

    def forward(self, cls_emb):
        x = torch.relu(self.fc1(cls_emb))
        x = self.dropout(x)
        return self.fc2(x)  # logits

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
def compute_token_accuracy(
    logits: torch.Tensor, 
    labels: torch.Tensor
) -> float:
    """
    Compute masked-LM token accuracy.
    logits: (B, L, V), labels: (B, L) with -100 for non-masked.
    """
    preds = logits.argmax(dim=-1)
    mask  = labels != -100
    if not mask.any():
        return 0.0
    return (preds[mask] == labels[mask]).float().mean().item()

def build_vocab(
    csv_path: Path, 
    train_subjects: List[str]
) -> Tuple[dict, List[str], int, int, int]:
    """
    Builds (token_to_id, id_to_token), plus pad_id, mask_id, vocab_size
    from the training subjects.
    """
    base_ds = TokenizedVCFDataset(
        csv_path=csv_path,
        subject_list=train_subjects,
        max_variants=None
    )
    token_to_id = base_ds.token_to_id
    id_to_token = [None] * len(token_to_id)
    for tok, idx in token_to_id.items():
        id_to_token[idx] = tok

    pad_id     = token_to_id["[PAD]"]
    mask_id    = token_to_id["[MASK]"]
    vocab_size = len(token_to_id)
    return token_to_id, id_to_token, pad_id, mask_id, vocab_size

def make_chunked_loaders(
    csv_path: Path,
    splits: Tuple[List[str], List[str], List[str]],
    token_to_id: dict,
    max_len: int,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader) of ChunkedVCFDataset.
    """
    train_subj, val_subj, test_subj = splits

    # Base DS for train (to build/update vocab)
    train_base = TokenizedVCFDataset(csv_path, train_subj, max_variants=None)
    train_ds   = ChunkedVCFDataset(train_base, max_len)

    # Val/test reuse the same vocab
    val_base   = TokenizedVCFDataset(csv_path, val_subj,   max_variants=None, token_to_id=token_to_id)
    test_base  = TokenizedVCFDataset(csv_path, test_subj,  max_variants=None, token_to_id=token_to_id)
    val_ds     = ChunkedVCFDataset(val_base, max_len)
    test_ds    = ChunkedVCFDataset(test_base, max_len)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )

def get_model_and_optimizer(
    vocab_size: int,
    max_len: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-2
):
    """
    Instantiates BigBirdForMaskedLM, AdamW, and returns them.
    """
    config = BigBirdConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=max_len,
        block_size=32,
        num_random_blocks=3,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        type_vocab_size=1,
        output_hidden_states=True
    )
    model     = BigBirdForMaskedLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer

def load_diagnosis(csv_path: Path):
    """
    Reads 'diagnosis.csv', strips parentheses, coerces to numeric,
    drops any NaNs, and converts to integer labels.
    """
    df = pd.read_csv(csv_path)
    df['PTID'] = df['PTID'].astype(str).str.strip('()')

    df['DIAGNOSIS'] = pd.to_numeric(
    df['DIAGNOSIS'].astype(str).str.strip('()'),
        errors='coerce'
    )
    df = df.dropna(subset=['DIAGNOSIS'])
    df['DIAGNOSIS'] = df['DIAGNOSIS'].astype(int) - 1     # ← shift to 0-based


    return dict(zip(df['PTID'], df['DIAGNOSIS']))