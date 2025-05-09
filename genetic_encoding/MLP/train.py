import math
import random
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from utils import (
    TokenizedVCFDataset,
    ChunkedVCFDataset,
    mask_inputs,
    build_vocab,
    make_chunked_loaders,
    compute_token_accuracy,
    PatientClassifier,
    get_model_and_optimizer,
    load_diagnosis
)

def main():
    # — Settings —
    data_csv      = Path('sample1.csv')
    subjects_csv  = Path('subjects.csv')
    diagnosis_csv = Path('diagnosis.csv')
    checkpoint    = Path('checkpoint.pt')

    epochs       = 5
    batch_size   = 128
    num_workers  = 0
    max_len      = 1024
    mask_probs   = { 'GT_': 0.25, 'GQ_': 0.25, 'PLMODE_': 0.25 }
    lr           = 1e-4
    weight_decay = 1e-2

    # — Load and clean diagnosis labels —
    label_map = load_diagnosis(diagnosis_csv)

    # — Load subjects and split —
    df_sub   = pd.read_csv(subjects_csv)
    subjects = list(df_sub.columns)
    random.seed(42)
    random.shuffle(subjects)

    n = len(subjects)
    train_subj = subjects[:int(0.7 * n)]
    val_subj   = subjects[int(0.7 * n):int(0.85 * n)]
    test_subj  = subjects[int(0.85 * n):]

    valid_ptids = set(label_map.keys())
    train_subj = [pt for pt in train_subj if pt in valid_ptids]
    val_subj   = [pt for pt in val_subj   if pt in valid_ptids]
    test_subj  = [pt for pt in test_subj  if pt in valid_ptids]

    # — Build vocab on training subjects —
    token_to_id, id_to_token, pad_id, mask_id, vocab_size = build_vocab(
        data_csv,
        train_subj
    )

    # — Prepare DataLoaders —
    train_loader, val_loader, test_loader = make_chunked_loaders(
        data_csv,
        (train_subj, val_subj, test_subj),
        token_to_id,
        max_len,
        batch_size,
        num_workers
    )

    # — Model and optimizer setup —
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer = get_model_and_optimizer(vocab_size, max_len, lr, weight_decay)
    clf_head = PatientClassifier(hidden_size=model.config.hidden_size)
    model.to(device)
    clf_head.to(device)

    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    clf_loss_fn = nn.CrossEntropyLoss()
    scaler      = torch.GradScaler(device="cpu")
    model.gradient_checkpointing_enable()

    # — Scheduler: 10% warmup + cosine decay —
    total_steps  = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            float(step) / warmup_steps if step < warmup_steps else
            0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        )
    )

    # — Resume checkpoint if it exists —
    global_step = 0
    best_val_loss = float('inf')
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        global_step   = ckpt['global_step'] + 1
        best_val_loss = ckpt['best_val_loss']

    # — Training loop —
    for epoch in range(1, epochs + 1):
        model.train()
        clf_head.train()
        train_mlm_loss = 0.0
        train_clf_loss = 0.0

        for batch_tokens, batch_ptids in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
            batch_tokens = batch_tokens.to(device)

            # Mask inputs for MLM
            masked_inputs, mlm_labels = mask_inputs(
                batch_tokens, token_to_id, id_to_token, mask_probs
            )
            attention_mask = (masked_inputs != pad_id).long().to(device)

            # Prepare classification labels from PTIDs
            diag_labels = torch.tensor(
                [label_map[ptid] for ptid in batch_ptids],
                dtype=torch.long,
                device=device
            )

            optimizer.zero_grad()
            with torch.autocast(device_type="cpu"):
                # MLM forward
                outputs     = model(
                    input_ids=masked_inputs,
                    attention_mask=attention_mask
                )
                mlm_logits  = outputs.logits  # (B, L, V)

                # Classification from CLS embedding
                cls_emb = outputs.hidden_states[-1][:, 0, :]  # (B, H)
                clf_logits  = clf_head(cls_emb) # (B, C)

                # Losses
                mlm_loss = mlm_loss_fn(
                    mlm_logits.view(-1, vocab_size),
                    mlm_labels.view(-1)
                )
                clf_loss = clf_loss_fn(clf_logits, diag_labels)

                # Joint loss: balance with alpha
                loss = mlm_loss + 0.5 * clf_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_mlm_loss += mlm_loss.item()
            train_clf_loss += clf_loss.item()
            global_step     += 1

        avg_mlm_loss = train_mlm_loss / len(train_loader)
        avg_clf_loss = train_clf_loss / len(train_loader)
        print(f"Epoch {epoch:02d}  MLM Loss {avg_mlm_loss:.4f}  CLF Loss {avg_clf_loss:.4f}")

        # — Validation —
        model.eval()
        clf_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_tokens, batch_ptids in tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]"):
                batch_tokens = batch_tokens.to(device)
                masked_inputs, mlm_labels = mask_inputs(
                    batch_tokens, token_to_id, id_to_token, mask_probs
                )
                attention_mask = (masked_inputs != pad_id).long().to(device)

                outputs     = model(
                    input_ids=masked_inputs,
                    attention_mask=attention_mask
                )
                mlm_logits  = outputs.logits
                cls_emb = outputs.hidden_states[-1][:, 0, :]
                clf_logits  = clf_head(cls_emb)

                mlm_loss = mlm_loss_fn(
                    mlm_logits.view(-1, vocab_size),
                    mlm_labels.view(-1)
                )
                diag_labels = torch.tensor(
                    [label_map[ptid] for ptid in batch_ptids],
                    dtype=torch.long,
                    device=device
                )
                clf_loss = clf_loss_fn(clf_logits, diag_labels)

                val_loss += (mlm_loss.item() + 0.5 * clf_loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch:02d}  Val Loss {avg_val_loss:.4f}")

        # — Checkpoint best —
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'global_step': global_step,
                'best_val_loss': best_val_loss
            }, checkpoint)

    # — Test evaluation —
    # ------ Test evaluation ------
    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state'])
    model.eval()
    clf_head.eval()

    test_loss    = 0.0
    test_clf_corr = 0
    test_clf_tot  = 0

    with torch.no_grad():
        for batch_tokens, batch_ptids in tqdm(test_loader, desc="[Test]"):
            batch_tokens = batch_tokens.to(device)

            # Mask MLM inputs
            masked_inputs, mlm_labels = mask_inputs(
                batch_tokens, token_to_id, id_to_token, mask_probs
            )
            attention_mask = (masked_inputs != pad_id).long().to(device)

            # Forward
            outputs    = model(input_ids=masked_inputs, attention_mask=attention_mask)
            mlm_logits = outputs.logits
            cls_emb    = outputs.hidden_states[-1][:,0,:]   # or use last_hidden_state if available
            clf_logits = clf_head(cls_emb)

            # Losses
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, vocab_size), mlm_labels.view(-1))
            diag_labels = torch.tensor(
                [label_map[ptid] for ptid in batch_ptids],
                dtype=torch.long,
                device=device
            )
            clf_loss = clf_loss_fn(clf_logits, diag_labels)

            test_loss += (mlm_loss.item() + 0.5*clf_loss.item())

            # Classification accuracy
            preds = clf_logits.argmax(dim=1)
            test_clf_corr += (preds == diag_labels).sum().item()
            test_clf_tot  += diag_labels.size(0)

    # Averages
    avg_test_loss = test_loss / len(test_loader)
    clf_accuracy  = test_clf_corr / test_clf_tot

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Test Classification Accuracy: {clf_accuracy:.2%}")

if __name__ == '__main__':
    main()
