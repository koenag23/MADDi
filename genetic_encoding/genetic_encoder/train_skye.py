import random
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils import (
    VCFDataset,
    mask_inputs,
    PatientClassifier,
    get_model_and_optimizer,
    load_diagnosis,
    cache_embeddings
)

# Enable CuDNN autotuner for fixed input sizes
torch.backends.cudnn.benchmark = True

# Configure logging to file
log_file = 'training.log'
logging.basicConfig(
    filename=log_file,
    filemode='a',  # append mode
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def save_checkpoint(model_path: Path, state: dict):
    torch.save(state, model_path)
    logger.info(f"Saved checkpoint: {model_path}")


def load_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    logger.info(f"Loaded checkpoint: {path}")
    return ckpt


def main():
    # — Settings —
    data_csv      = Path('sample_1000.csv')
    subjects_csv  = Path('subjects.csv')
    diagnosis_csv = Path('diagnosis.csv')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    embed_dim     = 768
    mlm_epochs    = 5
    clf_epochs    = 5
    batch_size    = 32
    num_workers   = 32
    seq_len       = 512
    total_snps    = 180000
    mask_probs    = {
        "REF_": 0.15,
        "ALT_": 0.15,
        "QUAL_": 0.2,
        "GT_":  0.3,
        "GQ_":  0.3,
        "PLMODE_": 0.3
    }
    lr            = 3e-4
    weight_decay  = 1e-2
    warmup_ratio  = 0.1

    # — Load labels and subjects —
    label_map = load_diagnosis(diagnosis_csv)
    df_sub    = pd.read_csv(subjects_csv)
    subjects  = list(df_sub.columns)
    random.seed(42); random.shuffle(subjects)
    n = len(subjects)
    split1, split2 = int(0.7*n), int(0.85*n)
    train_subj = [pt for pt in subjects[:split1] if pt in label_map]
    '''
    val_subj   = [pt for pt in subjects[split1:split2] if pt in label_map]
    test_subj  = [pt for pt in subjects[split2:] if pt in label_map]
    '''
    # — Build datasets —
    train_ds = VCFDataset(
        data_csv, subject_list=train_subj,
        seq_len=seq_len, total_snps=total_snps
    )
    token_to_id = train_ds.token_to_id
    id_to_token = train_ds.id_to_token
    pad_id      = token_to_id[train_ds.pad_token]
    vocab_size  = len(token_to_id)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, 
        persistent_workers= True
    )
    
    '''
    val_ds  = VCFDataset(
        data_csv, subject_list=val_subj,
        seq_len=seq_len, total_snps=total_snps,
        token_to_id=token_to_id
    )
    test_ds = VCFDataset(
        data_csv, subject_list=test_subj,
        seq_len=seq_len, total_snps=total_snps,
        token_to_id=token_to_id
    )

    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    '''
    # — Initialize model & optimizer —
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer = get_model_and_optimizer(
        vocab_size, embed_dim=embed_dim, seq_len=seq_len,
        lr=lr, weight_decay=weight_decay
    )
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_steps  = mlm_epochs * 8000
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.GradScaler(device="cuda")

    # — MLM Pretraining Loop —
    for epoch in range(1, mlm_epochs+1):
        model.train()
        epoch_loss = 0.0
        for _, (tokens, _) in tqdm(
            enumerate(train_loader), desc=f"MLM Epoch {epoch}/{mlm_epochs}"
        ):
            tokens = tokens.to(device, non_blocking=True)
            masked, labels = mask_inputs(
                tokens, token_to_id, id_to_token, mask_probs
            )
            pad_mask = (masked == pad_id).to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, logits = model(masked, attention_mask=pad_mask)
                loss = mlm_loss_fn(
                    logits.view(-1, vocab_size), labels.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} MLM loss: {avg_loss:.4f}")
        # save main model each epoch
        ckpt_path = checkpoint_dir / f"mlm_epoch{epoch}.pt"
        save_checkpoint(ckpt_path, {'model_state_dict': model.state_dict()})
    '''
    # — Cache embeddings for downstream classifier —
    train_emb, train_lab = cache_embeddings(
        model, train_loader, device, pad_id, label_map,
        mask_probs, token_to_id, id_to_token
    )
    val_emb, val_lab = cache_embeddings(
        model, val_loader, device, pad_id, label_map,
        mask_probs, token_to_id, id_to_token
    )
    test_emb, test_lab = cache_embeddings(
        model, test_loader, device, pad_id, label_map,
        mask_probs, token_to_id, id_to_token
    )

    # — Train classifier on cached embeddings —
    clf_head = PatientClassifier(hidden_size=embed_dim)
    clf_head.to(device)
    opt2 = torch.optim.AdamW(
        clf_head.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_ds_clf = torch.utils.data.TensorDataset(train_emb, train_lab)
    test_ds_clf  = torch.utils.data.TensorDataset(test_emb, test_lab)
    train_loader2 = DataLoader(
        train_ds_clf, batch_size=batch_size, shuffle=True
    )
    test_loader2  = DataLoader(test_ds_clf,  batch_size=batch_size)

    for epoch in range(1, clf_epochs+1):
        clf_head.train()
        total, correct = 0, 0
        for emb, labs in tqdm(
            train_loader2, desc=f"CLF Epoch {epoch}/{clf_epochs}"
        ):
            emb, labs = emb.to(device), labs.to(device)
            opt2.zero_grad()
            logits = clf_head(emb)
            loss   = nn.CrossEntropyLoss()(logits, labs)
            loss.backward(); opt2.step()
            preds = logits.argmax(dim=1)
            correct += (preds==labs).sum().item(); total += labs.size(0)
        acc = correct/total
        logger.info(f"CLF Epoch {epoch} acc: {acc:.2%}")

    # — Final Test Accuracy —
    clf_head.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for emb, labs in test_loader2:
            emb, labs = emb.to(device), labs.to(device)
            preds = clf_head(emb).argmax(dim=1)
            correct += (preds==labs).sum().item(); total += labs.size(0)
    logger.info(f"Test accuracy: {correct/total:.2%}")
    '''
if __name__ == '__main__':
    main()