import pandas as pd
from pathlib import Path
import bitsandbytes as bnb
import torch
import torch.nn as nn
from model import BERT
from torch.utils.data import Dataset
import torch.distributed as dist

class VCFDataset(Dataset):
    """
    Combines tokenization, vocab building, and fixed-length chunk streaming per patient.
    """
    def __init__(
        self,
        csv_path: Path,
        subject_list: list[str] | None = None,
        max_len: int = 1024,
        format_fields: tuple[str, ...] = ("GT", "GQ", "PLMODE"),
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        mask_token: str = "[MASK]",
        token_to_id: dict[str,int] | None = None,
        name: str = None
    ):
        # load data
        self.df = pd.read_csv(csv_path)
        fixed = ["#CHROM","ALT","FILTER","FORMAT","ID","INFO","POS","QUAL","REF"]
        all_subjects = [c for c in self.df.columns if c not in fixed]
        self.subjects = subject_list or all_subjects
        self.name = name

        # tokens
        self.format_fields = format_fields
        self.max_len = max_len

        # vocab
        self.pad_token, self.unk_token, self.mask_token = pad_token, unk_token, mask_token
        self.token_to_id = token_to_id or {pad_token:0, unk_token:1, mask_token:2}
        self.id_to_token = list(self.token_to_id.keys())
        self._build_sequences()

    def _build_sequences(self):
        self.subject_sequences = {}
        self.data_index = []  # index of (subject, chunk_start_idx) for __getitem__
        
        update_vocab = True
        if len(self.id_to_token) > 3:
            update_vocab = False

        for subj in self.subjects:
            tokens = []
            for i, cell in enumerate(self.df[subj].fillna("0/0:.:.:.:.")):
                parts = str(cell).split(":")
                # GT
                gt = parts[0]
                tokens += ["GT_", f"##{gt}"]
                # GQ
                if len(parts) >= 4:
                    try:
                        gq = int(parts[3])
                        b = f"{(gq // 10) * 10}-{((gq // 10) + 1) * 10}"
                        tokens += ["GQ_", f"##{b}"]
                    except:
                        tokens.append(self.unk_token)
                # PLMODE
                if len(parts) >= 5:
                    pl = [int(p) for p in parts[4].split(',') if p.isdigit()]
                    if pl:
                        m = pl.index(min(pl))
                        tokens += ["PLMODE_", f"##{m}"]
                tokens.append("[SEP]")

            self.subject_sequences[subj] = tokens

            # Update vocab if needed
            if update_vocab:
                for t in tokens:
                    if t not in self.token_to_id:
                        self.token_to_id[t] = len(self.token_to_id)
                        self.id_to_token.append(t)

            # Track chunks for __getitem__
            for start_idx in range(0, len(tokens), self.max_len):
                self.data_index.append((subj, start_idx))

    def __getitem__(self, index):
        pad_id = self.token_to_id[self.pad_token]
        unk_id = self.token_to_id[self.unk_token]

        ptid, start_idx = self.data_index[index]
        toks = self.subject_sequences[ptid]
        ids = [self.token_to_id.get(t, unk_id) for t in toks]
        chunk = ids[start_idx:start_idx + self.max_len]
        if len(chunk) < self.max_len:
            chunk += [pad_id] * (self.max_len - len(chunk))
        return torch.tensor(chunk, dtype=torch.long), ptid

    def __len__(self):
        return len(self.data_index)

# 3) Simple 2‐layer head for classification
class PatientClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes=3, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.do  = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size//2, num_classes)

    def forward(self, cls_emb):
        x = torch.relu(self.fc1(cls_emb))
        x = self.do(x)
        return self.fc2(x)  # logits

def mask_inputs(x, MASKABLE_BOOL, prob_map, id_to_token, token_to_id):
    B, L = x.shape
    device = x.device

    # 1) Build a boolean mask of all “value” tokens (those starting with '##')
    # maskable = MASKABLE_BOOL[x]
    device = MASKABLE_BOOL.device
    maskable = x.to(device)

    # 2) Build a vector `prob_map` so prob_map[field_id] == mask probability

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
    
    keep = torch.rand(b_idxs.size(0), device=device)
    
    mask80 = keep < 0.8
    x_masked[b_idxs[mask80], pos_idxs[mask80] + 1] = token_to_id["[MASK]"]
    # 10% → random
    mask10 = ~mask80 & (keep < 0.1)
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

def get_model_and_optimizer(
    vocab_size: int,
    embed_dim: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-2
):
    """
    Instantiates BigBirdForMaskedLM, AdamW, and returns them.
    """
    model = BERT(vocab_size=vocab_size,
                 embed_dim=embed_dim,
                 num_layers=8,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 use_flash_attention=True)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
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

def log_txt(log_path: str, data):
    """
    Logs output into the log file
    """
    
    with open(log_path, "w") as f:
        f.write(str(data))