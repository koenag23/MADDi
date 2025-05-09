import pandas as pd
from pathlib import Path
import bitsandbytes as bnb
import torch
import torch.nn as nn
from model_skye import Bert
import pandas as pd
from torch.utils.data import IterableDataset

class VCFDataset(IterableDataset):
    """
    Tokenizes VCF SNP data with enriched metadata tokens and supports field-aware masking:
      - Chromosome & binned position
      - SNP ID
      - REF/ALT alleles (field_ + ##value)
      - Variant type (field_ + ##value)
      - FILTER flag (field_ + ##value)
      - QUAL as raw value (field_ + ##value)
      - Optional region annotation
      - Format fields (GT, GQ, PLMODE) (field_ + ##value)
    Missing entries are tokenized as field_ + ##None to keep sequence lengths normalized.
    """
    def __init__(
        self,
        csv_path: Path,
        subject_list: list[str] | None = None,
        seq_len: int = 1024,
        total_snps: int | None = None,
        format_fields: tuple[str, ...] = ("GT", "GQ", "PLMODE"),
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        token_to_id: dict[str,int] | None = None,
        region_map: dict[str,str] | None = None,
        chrom_binsize: int = 1_000_000,
    ):
        fixed = ["#CHROM","ALT","FILTER","FORMAT","ID","INFO","POS","QUAL","REF"]
        if subject_list:
            self.subjects = pd.read_csv(csv_path, usecols=subject_list, dtype='string')
            self.subject_list = subject_list
        else:
            self.subjects = pd.read_csv(csv_path, usecols=lambda col: col[0].isnumeric(), dtype='string')
            self.subject_list = self.subjects.columns
        self.metadata = pd.read_csv(csv_path, usecols=fixed)

        self.seq_len        = seq_len
        self.total_snps     = total_snps
        self.format_fields  = format_fields
        self.region_map     = region_map or {}
        self.chrom_binsize  = chrom_binsize

        self.pad_token, self.mask_token = pad_token, mask_token
        # initialize vocab with pad and mask
        self.token_to_id = token_to_id or {pad_token:0, mask_token:1}
        self.id_to_token = list(self.token_to_id.keys())

        self._build_sequences()

    def _build_sequences(self):
        self.subject_sequences = {}
        update_vocab = len(self.id_to_token) <= 2 
        metadata = self.metadata.to_numpy()
        
        for subj in self.subject_list:
            tokens = []
            cells = self.subjects[subj].fillna('None')
            if self.total_snps is not None:
                cells = cells[:self.total_snps]
                
            cells = cells.to_numpy()
            for idx in range(metadata.shape[0]):
                cell = cells[idx]
                row = metadata[idx]
                # extract metadata fields
                
                chrom = row[0]
                alt   = row[1]
                snp_id= row[4]
                pos   = row[6]
                ref   = row[8]
                qual  = row[7]

                # CHROM and POS
                if chrom is not None:
                    tokens.append(f"CHR_{chrom}")
                tokens.append(
                    f"POS_{int(pos//self.chrom_binsize)}Mb"
                    if pd.notna(pos) else "POS_##None"
                )
                # SNP ID
                if snp_id is not None:
                    tokens.append(f"ID_{snp_id}")
                # REF/ALT
                tokens += [
                    "REF_", f"##{ref or 'None'}",
                    "ALT_", f"##{alt or 'None'}"
                ]
                
                # QUAL
                tokens += ["QUAL_", f"##{int(qual) if pd.notna(qual) else 'None'}"]
                
                # FORMAT fields: GT, GQ, PLMODE
                parts = str(cell).split(":") if cell is not None else []
                gt = parts[0] if len(parts)>=1 and parts[0] not in ('.','') else 'None'
                gq = parts[3] if len(parts)>=4 and parts[3].isdigit() else 'None'
                plm = (
                    str(min((int(p) for p in parts[4].split(",") if p.isdigit()), default='None'))
                    if len(parts)>=5 else 'None'
                )
                tokens += [
                    "GT_", f"##{gt}",
                    "GQ_", f"##{gq}",
                    "PLMODE_", f"##{plm}"
                ]
                tokens.append("[SEP]")

            self.subject_sequences[subj] = tokens
            if update_vocab:
                for t in tokens:
                    if t not in self.token_to_id:
                        self.token_to_id[t] = len(self.token_to_id)
                        self.id_to_token.append(t)

    def __iter__(self):
        pad_id = self.token_to_id[self.pad_token]
        for ptid in self.subject_list:
            ids = [self.token_to_id.get(tok, pad_id) for tok in self.subject_sequences[ptid]]
            for i in range(0, len(ids), self.seq_len):
                chunk = ids[i:i+self.seq_len]
                if len(chunk)<self.seq_len:
                    chunk += [pad_id]*(self.seq_len-len(chunk))
                yield torch.tensor(chunk, dtype=torch.long), ptid

    """ def __len__(self):
        total=0
        for toks in self.subject_sequences.values():
            total+=(len(toks)+self.seq_len-1)//self.seq_len
        return total """
    
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

def get_model_and_optimizer(
    vocab_size: int,
    embed_dim: int,
    seq_len: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-2
):
    """
    Instantiates BigBirdForMaskedLM, AdamW, and returns them.
    """
    model = Bert(vocab_size=vocab_size,
                 embed_dim=embed_dim,
                 seq_len=seq_len,
                 num_layers=12,
                 num_heads=12,
                 dim_feedforward=3072,
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

def count_unks(ds: VCFDataset):
    unk_id = ds.token_to_id[ds.unk_token]
    total_tokens = 0
    unk_tokens   = 0

    for chunk, ptid in ds:
        # chunk is a LongTensor of shape (seq_len,)
        total_tokens += chunk.numel()
        unk_tokens   += (chunk == unk_id).sum().item()

    return total_tokens, unk_tokens

def cache_embeddings(
    model, loader, device, pad_id, label_map,
    mask_probs, token_to_id, id_to_token
):
    """
    Run data through the frozen MLM model, extract CLS embeddings.
    Applies masking using provided mask_inputs arguments.
    """
    model.eval()
    all_embs, all_labels = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for tokens, ptids in loader:
            tokens = tokens.to(device)
            masked, _ = mask_inputs(
                tokens, token_to_id, id_to_token, mask_probs
            )
            pad_mask = (masked == pad_id).to(device)
            seq_emb, _ = model(masked, attention_mask=pad_mask)
            cls_emb = seq_emb[:, 0, :].cpu()
            labels = torch.tensor(
                [label_map[pt] for pt in ptids],
                dtype=torch.long
            )
            all_embs.append(cls_emb)
            all_labels.append(labels)
    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)