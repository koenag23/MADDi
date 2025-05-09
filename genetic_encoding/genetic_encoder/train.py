import random
import gc
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import (
    VCFDataset,
    mask_inputs,
    PatientClassifier,
    get_model_and_optimizer,
    load_diagnosis,
    compute_token_accuracy,
    log_txt
)

torch.backends.cudnn.benchmark = True

def main():
    # — Settings —
    data_csv      = Path('sample_1000.csv')
    subjects_csv  = Path('subjects.csv')
    diagnosis_csv = Path('diagnosis.csv')
    checkpoint    = Path('checkpoint.pt')

    embed_dim     = 512
    mlm_epochs    = 25
    clf_epochs    = 100
    batch_size    = 64
    num_workers   = 32
    max_len       = 2048
    mask_probs    = {'GT_':0.25, 'GQ_':0.25, 'PLMODE_':0.25}
    lr            = 1e-4
    weight_decay  = 1e-2
    alpha         = 0.5
    seed          = 42
    
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # - Set Random Seeds -
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # — Load labels and subjects —
    label_map   = load_diagnosis(diagnosis_csv)
    df_sub      = pd.read_csv(subjects_csv)
    subjects    = list(df_sub.columns)
    random.shuffle(subjects)
    n = len(subjects)
    train_subj  = subjects[:int(0.7*n)]
    val_subj    = subjects[int(0.7*n):int(0.85*n)]
    test_subj   = subjects[int(0.85*n):]
    valid       = set(label_map.keys())
    train_subj  = [pt for pt in train_subj if pt in valid]
    val_subj    = [pt for pt in val_subj   if pt in valid]
    test_subj   = [pt for pt in test_subj  if pt in valid]

    # — Build vocab & datasets —
    base_ds = VCFDataset(data_csv, subject_list=train_subj, max_len=max_len, name="Train")
    # share vocab for val/test
    token_to_id, id_to_token, pad_id, vocab_size = base_ds.token_to_id, base_ds.id_to_token, base_ds.token_to_id[base_ds.pad_token], len(base_ds.token_to_id)
    
    MASKABLE_BOOL = torch.zeros(len(id_to_token), dtype=torch.bool, device=device)
    for idx, tok in enumerate(id_to_token):
        if tok.startswith("##"):
            MASKABLE_BOOL[idx] = True
    
    prob_map = torch.zeros(len(id_to_token), device=device)
    for field, p in mask_probs.items():
        prob_map[token_to_id[field]] = p
    
    val_ds  = VCFDataset(data_csv, subject_list=val_subj, max_len=max_len, token_to_id=token_to_id, name="Val")
    test_ds = VCFDataset(data_csv, subject_list=test_subj, max_len=max_len, token_to_id=token_to_id, name="Test")

    # — DataLoaders —
    train_loader = DataLoader(base_ds,   batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,    batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,   batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # — Model & optimizer —
    model, optimizer = get_model_and_optimizer(vocab_size, embed_dim=embed_dim, lr=lr, weight_decay=weight_decay)
    clf_head         = PatientClassifier(hidden_size=embed_dim)
    model.to(device); clf_head.to(device)
    
    if torch.cuda.device_count() > 1:
        model.bert = nn.DataParallel(model.bert)

    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    clf_loss_fn = nn.CrossEntropyLoss()
    scaler      = torch.GradScaler(device="cuda")
    
    # optimizer for clf head only
    clf_head_opt = torch.optim.AdamW(clf_head.parameters(), lr=lr, weight_decay=weight_decay)
    # simple scheduler
    clf_head_sch = LambdaLR(clf_head_opt, lr_lambda=lambda s: 1.0)
    
    # — Cache CLS embeddings for all splits —
    def cache_embeddings(loader):
        embs = []
        labs = []
        loader_name = getattr(loader.dataset, 'name', 'dataset')
        e_loader = tqdm(enumerate(loader), position=2, leave=False, desc=f"Caching {loader_name}")
        for _, (tokens, ptids) in e_loader:
            tokens = tokens.to(device, non_blocking=True)
            with torch.no_grad():
                masked, _ = mask_inputs(tokens, MASKABLE_BOOL, prob_map, id_to_token, token_to_id)
                pad_mask = (masked==pad_id).to(device)
                seq_emb = model.bert(masked, attention_mask=pad_mask)
            cls_emb = seq_emb[:,0,:].cpu()
            idx = [label_map[pt] for pt in ptids]
            labels = torch.as_tensor(idx, dtype=torch.long)
            embs.append(cls_emb); labs.append(labels)
            del tokens, idx, labels
            gc.collect()
            torch.cuda.empty_cache()
        return torch.cat(embs, dim=0), torch.cat(labs, dim=0)

    # — Phase 1: MLM-only training —
    best_val_loss = float('inf')
    mlm_pbar = tqdm(range(1, mlm_epochs+1), position=0, desc="MLM")
    for epoch in mlm_pbar:
        model.train()
        mlm_loss_sum = 0.0
        stream_train_len = 0
        for _, (tokens, _) in tqdm(enumerate(train_loader), leave=False, desc="Training"):
            stream_train_len += tokens.shape[0]
            tokens = tokens.to(device, non_blocking=True)
            masked, labels = mask_inputs(tokens, MASKABLE_BOOL, prob_map, id_to_token, token_to_id)
            pad_mask = (masked==pad_id).to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                seq_emb = model.bert(masked, attention_mask=pad_mask)
                logits  = model.mlm_head(seq_emb)
                loss    = mlm_loss_fn(logits.view(-1, vocab_size), labels.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            mlm_loss_sum += loss.item()
        mlm_pbar.set_description(f"Epoch {epoch}")
        mlm_pbar.set_postfix(train_loss=mlm_loss_sum/stream_train_len)
        
        print(stream_train_len, train_loader.dataset.length)
            
        torch.cuda.empty_cache()        
        model.eval()

        val_loss = 0
        val_acc = 0
        stream_val_len = 0

        with torch.no_grad():
            for _, (tokens, _) in tqdm(enumerate(val_loader), leave=False, desc="Validation"):
                batch_size = tokens.shape[0]     
                stream_val_len += batch_size
                tokens = tokens.to(device, non_blocking=True)
                
                masked_tokens, labels = mask_inputs(tokens, MASKABLE_BOOL, prob_map, id_to_token, token_to_id)
                masked_tokens, labels = masked_tokens.to(device), labels.to(device)
                pad_mask = (masked_tokens==pad_id).to(device)

                embeddings = model.bert(masked_tokens, attention_mask=pad_mask)
                logits = model.mlm_head(embeddings)
                loss = mlm_loss_fn(logits.view(-1, logits.size(-1)), labels.reshape(-1))
                
                val_acc += compute_token_accuracy(logits, labels) * batch_size
                val_loss += loss.item() * batch_size

        avg_val_loss = val_loss / stream_val_len
        avg_val_acc  = val_acc  / stream_val_len

        tqdm.write(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")
            
        if avg_val_loss < best_val_loss:
            tqdm.write(f"Saving model at epoch {epoch}")
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint)
            
        torch.cuda.empty_cache()        
        cache_pbar = tqdm(total=3, position=1, leave=False, desc="Caching splits")
            
        train_emb, train_lab = cache_embeddings(train_loader); cache_pbar.update(1)
        val_emb, val_lab = cache_embeddings(val_loader); cache_pbar.update(1)
        test_emb, test_lab = cache_embeddings(test_loader); cache_pbar.update(1)
        cache_pbar.close()
        torch.cuda.empty_cache()
        
        # — Phase 2: train classifier on cached embeddings —
        clf_ds_train = TensorDataset(train_emb, train_lab)
        clf_ds_val   = TensorDataset(val_emb,   val_lab)
        clf_ds_test  = TensorDataset(test_emb,  test_lab)
        clf_loader   = DataLoader(clf_ds_train, batch_size=batch_size, shuffle=True)
        val_loader2  = DataLoader(clf_ds_val,   batch_size=batch_size)
        test_loader2 = DataLoader(clf_ds_test,  batch_size=batch_size)

        clf_pbar = tqdm(range(1, clf_epochs+1), leave=False, desc='CLF')
        for clf_epoch in clf_pbar:
            clf_head.train()
            sum_loss = corr = tot = 0
            for _, (emb, lab) in enumerate(clf_loader):
                emb, lab = emb.to(device), lab.to(device)
                clf_head_opt.zero_grad()
                logits = clf_head(emb)
                loss   = clf_loss_fn(logits, lab)
                loss.backward(); clf_head_opt.step(); clf_head_sch.step()
                sum_loss += loss.item()
                preds = logits.argmax(dim=1)
                corr += (preds==lab).sum().item(); tot += lab.size(0)
            acc = corr/tot
            clf_pbar.set_description(f"CLF Epoch {clf_epoch}")
            clf_pbar.set_postfix(loss=sum_loss/len(clf_loader), acc=f"{acc:.2%}")

    # final test accuracy
    corr = tot = 0
    with torch.no_grad():
        clf_head.eval()
        for emb, lab in test_loader2:
            emb, lab = emb.to(device), lab.to(device)
            preds = clf_head(emb).argmax(dim=1)
            corr += (preds==lab).sum().item(); tot += lab.size(0)
    tqdm.write(f"Test clf acc: {corr/tot:.2%}")
    
    # save final encoder
    torch.save(model.state_dict(), checkpoint.with_suffix('.mlm.pt'))

if __name__ == '__main__':
    main()
