import torch
import torch.nn as nn
from utils_old import TokenizedVCFDataset, mask_inputs
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
from transformers import BigBirdConfig, BigBirdForMaskedLM
import math
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_embeddings(data_loader, model, token_to_id, id_to_token, field_mask_probs):
    embeddings = [] 
    for batch_tokens, _ in data_loader:
        with torch.no_grad():
            batch_tokens = batch_tokens.to(device)
            masked_inputs, labels = mask_inputs(batch_tokens, token_to_id, id_to_token, field_mask_probs)

            masked_inputs, labels = masked_inputs.to(device), labels.to(device)

            outputs = model(input_ids=masked_inputs, output_hidden_states=True)
            
            hidden = outputs.hidden_states[-1]
        
        cls_embeddings = hidden[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
        
    return np.vstack(embeddings)  
    

def compute_token_accuracy(logits, labels):
    """
    logits: (B, L, V)
    labels: (B, L), with -100 for unmasked
    """
    preds = logits.argmax(dim=-1)
    mask  = labels != -100
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0

def main():
    # 1) Load subject list and split
    df_sub = pd.read_csv('subjects.csv')
    subjects = list(df_sub.columns)
    random.seed(42)
    random.shuffle(subjects)

    n = len(subjects)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    train_subj = subjects[:n_train]
    val_subj   = subjects[n_train:n_train + n_val]
    test_subj  = subjects[n_train + n_val:]
    all_subj = subjects[:]

    # 2) Build / freeze vocabulary with the train split
    max_variants = 1024
    train_ds = TokenizedVCFDataset(
        csv_path="combined.csv",
        subject_list=train_subj,
        max_variants=max_variants  # use all or up to this many tokens
    )
    # Now freeze vocab
    token_to_id = train_ds.token_to_id
    id_to_token = [None] * len(token_to_id)
    for token, idx in token_to_id.items():
        id_to_token[idx] = token
    assert all(id_to_token[token_to_id[tok]] == tok for tok in token_to_id), "token ↔ ID mismatch"
    pad_id  = token_to_id["[PAD]"]
    mask_id = token_to_id["[MASK]"]
    vocab_size = len(token_to_id)

    # 3) Build val/test with the same vocab
    val_ds = TokenizedVCFDataset(
        csv_path="data.csv",
        subject_list=val_subj,
        max_variants=max_variants,
        token_to_id=token_to_id
    )
    test_ds = TokenizedVCFDataset(
        csv_path="data.csv",
        subject_list=test_subj,
        max_variants=max_variants,
        token_to_id=token_to_id
    )

    train_loader    = DataLoader(train_ds, batch_size=16)
    val_loader      = DataLoader(val_ds,   batch_size=16)
    test_loader     = DataLoader(test_ds,  batch_size=16)

    # 4) BigBird setup
    bb_config = BigBirdConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=max_variants,
        block_size=64,
        num_random_blocks=3,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        type_vocab_size=1,
    )
    model = BigBirdForMaskedLM(bb_config).to(device)

    # 2) Optimizer & Loss
    epochs       = 50
    warmup_steps = 350
    optimizer    = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn      = nn.CrossEntropyLoss(ignore_index=-100)

    # 3) LR schedule (warmup + cosine)
    def lr_lambda(step):
        # step is an integer (global step)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # once past warmup, do cosine annealing on the _fraction_ of total steps
        total_steps = epochs * len(train_loader)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # 4) Masking & AMP setup
    field_mask_probs = {"GT_":0.25, "GQ_":0.25, "PLMODE_":0.25}
    model.gradient_checkpointing_enable()
    scaler = torch.GradScaler("cuda")
    
    checkpoint_path = "checkpoint.pt"
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    
    epoch, avg_test_loss, avg_train_loss, avg_val_loss = 0, float('inf'), float('inf'), float('inf')
    
    pbar = tqdm(range(epochs), desc=f"Epoch {epoch+1} | "
                                    f"Train Loss: {avg_train_loss:.4f} | "
                                    f"Val Loss: {avg_val_loss:.4f}", )

    for epoch in pbar:
        # ------ Train ------
        model.train()
        running_loss = 0.0
        running_acc  = 0.0

        for batch_idx, (batch_tokens, _) in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                tokens = batch_tokens[0][:60].tolist()
                tqdm.write(f"Raw token IDs: {tokens}")
                tqdm.write(f"Tokens: {[id_to_token[i] for i in tokens]}")
            batch_tokens = batch_tokens.to(device)
            masked_inputs, labels = mask_inputs(batch_tokens, token_to_id, id_to_token, field_mask_probs)
            if epoch == 0 and batch_idx == 0:
                masked_inputs, labels = masked_inputs.to(device), labels.to(device)
                tokens = batch_tokens[0][:60].tolist()
                masked = masked_inputs[0][:60].tolist()
                label  = labels[0][:60].tolist()

                tqdm.write("\nInput → Masked → Label")
                for i, (orig, m, l) in enumerate(zip(tokens, masked, label)):
                    tok_orig = id_to_token[orig]
                    tok_mask = id_to_token[m] if 0 <= m < len(id_to_token) else "[UNK]"
                    tok_label = id_to_token[l] if l != -100 else "[IGNORE]"
                    tqdm.write(f"{i:02d}: {tok_orig:15} → {tok_mask:15} → {tok_label}")
            # ------ forward + backward with AMP ------
            with torch.autocast("cuda"):
                outputs = model(input_ids=masked_inputs)
                logits  = outputs.logits                     # (B, L, V)
                loss    = loss_fn(logits.view(-1, vocab_size),
                                labels.view(-1))

            acc = compute_token_accuracy(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # step scheduler by global step
            global_step += 1
            scheduler.step()

            running_loss += loss.item()
            running_acc  += acc

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc  = running_acc  / len(train_loader)

        # ------ Validation ------
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0

        with torch.no_grad():
            for batch_tokens, _ in val_loader:
                batch_tokens = batch_tokens.to(device)
                masked_inputs, labels = mask_inputs(batch_tokens, token_to_id, id_to_token, field_mask_probs)

                masked_inputs, labels = masked_inputs.to(device), labels.to(device)

                outputs = model(input_ids=masked_inputs)
                logits  = outputs.logits
                loss    = loss_fn(logits.view(-1, vocab_size),
                                labels.view(-1))
                acc     = compute_token_accuracy(logits, labels)

                val_loss += loss.item()
                val_acc  += acc

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = val_acc  / len(val_loader)

        pbar.set_description_str(f"Epoch {epoch+1}")
        pbar.set_postfix_str(f"Train Loss: {avg_train_loss:.4f} | "
                            f"Val Loss: {avg_val_loss:.4f}")
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        if epoch % 5 == 0 and epoch >= 5:
            tqdm.write(f"Saving model at epoch {epoch}")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_path)

    # ------ Test ------
    model.eval()
    test_loss = 0.0
    test_acc  = 0.0

    for batch_tokens, _ in tqdm(test_loader, desc=">>>> TEST"):
        with torch.no_grad():
            batch_tokens = batch_tokens.to(device)
            masked_inputs, labels = mask_inputs(batch_tokens, token_to_id, id_to_token, field_mask_probs)

            masked_inputs, labels = masked_inputs.to(device), labels.to(device)

            outputs = model(input_ids=masked_inputs, output_hidden_states=True)

            
            logits  = outputs.logits
            loss    = loss_fn(logits.view(-1, vocab_size),
                            labels.view(-1))
            acc     = compute_token_accuracy(logits, labels)

            test_loss += loss.item()
            test_acc  += acc
        

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc  = test_acc 
    print(f"\nTest Loss: {avg_test_loss:.4f}, Accuracy: {100*avg_test_acc:.2f}%")
    
    train_embeddings    = extract_embeddings(train_loader, model, token_to_id, id_to_token, field_mask_probs)
    val_embeddings      = extract_embeddings(val_loader, model, token_to_id, id_to_token, field_mask_probs)
    test_embeddings     = extract_embeddings(test_loader, model, token_to_id, id_to_token, field_mask_probs)
    
    print(f'Shape of Train Latent Vector Space: {train_embeddings.shape}')
    print(f'Shape of Val Latent Vector Space: {val_embeddings.shape}')
    print(f'Shape of Test Latent Vector Space: {test_embeddings.shape}')
    
    np.save("train_embeddings.npy", train_embeddings)
    np.save("val_embeddings.npy", val_embeddings)
    np.save("test_embeddings.npy", test_embeddings)

if __name__ == '__main__':
    main()
