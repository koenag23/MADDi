import torch
import torch.nn as nn
from utils import TokenizedVCFDataset, MinimalBERT, mask_inputs
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_token_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()

def print_tokenization_example(dataset, subject_index=0):
    # Print the tokenization of a single subject using available tokens.
    subj = dataset.subjects[subject_index]
    tokens = dataset.subject_sequences[subj]
    num_variants = len(tokens) // len(dataset.format_fields)
    print(f"\n--- Tokenization for Subject: {subj} (showing {min(num_variants, 10)} variants) ---")
    for i in range(min(num_variants, 10)):
        start = i * len(dataset.format_fields)
        end = start + len(dataset.format_fields)
        print(f"[{i}] Tokens: {tokens[start:end]}")

def main():
    # ---- Load Data and Create Subject Split ----
    subjects = pd.read_csv('subjects.csv')
    subjects = [col for col in subjects.columns]
    random.seed(42)
    random.shuffle(subjects)

    train_size = int(0.7 * len(subjects))
    val_size = int(0.15 * len(subjects))
    # Ensure the remaining subjects are used for testing.
    train_subjects = subjects[:train_size]
    val_subjects = subjects[train_size:train_size + val_size]
    test_subjects = subjects[train_size + val_size:]

    # ---- Build Vocabulary and Set Special Tokens ----
    special_tokens = ["[PAD]", "[UNK]", "[MASK]"]
    pad_token, unk_token, mask_token = "[PAD]", "[UNK]", "[MASK]"
    token_to_id = {pad_token: 0, unk_token: 1, mask_token: 2}

    for tok in special_tokens:
        token_to_id.setdefault(tok, len(token_to_id))

    pad_id = token_to_id[pad_token]
    mask_id = token_to_id[mask_token]
    vocab_size = len(token_to_id)

    # ---- Create Datasets with Shared Vocabulary ----
    common_args = dict(csv_path="sample.csv", max_variants=1024, use_subtoken_strategy=True)
    train_dataset = TokenizedVCFDataset(subject_list=train_subjects, **common_args)  # ✅ Let it build its own vocab

    token_to_id = train_dataset.token_to_id
    pad_id = token_to_id["[PAD]"]
    mask_id = token_to_id["[MASK]"]
    vocab_size = len(token_to_id)

    # Use the frozen vocab for validation and test
    val_dataset = TokenizedVCFDataset(subject_list=val_subjects, token_to_id=token_to_id, **common_args)             
    test_dataset = TokenizedVCFDataset(subject_list=test_subjects, token_to_id=token_to_id, **common_args)
                                                            
    vocab_size = len(train_dataset.token_to_id)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8)
    test_loader  = DataLoader(test_dataset, batch_size=8)

    # ---- Model and Training Setup ----
    model = MinimalBERT(vocab_size=vocab_size, embed_dim=768, nhead=12, num_layers=12, max_len=6144).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(50):
        model.train()
        train_loss = 0.0

        for batch_tokens, _ in train_loader:

            batch_tokens = batch_tokens.to(device)
            masked_inputs, labels = mask_inputs(batch_tokens, vocab_size, mask_id, pad_id)
            logits = model(masked_inputs)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_tokens, _ in val_loader:
                val_tokens = val_tokens.to(device)
                masked_inputs, labels = mask_inputs(val_tokens, vocab_size, mask_id, pad_id)
                logits = model(masked_inputs)
                loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()  # StepLR requires no argument.
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    # ---- Load Best Model and Test Evaluation ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    test_loss = 0.0
    total_correct = 0
    total_masked = 0

    with torch.no_grad():
        for test_tokens, _ in tqdm(test_loader, desc="Evaluating on Test"):
            test_tokens = test_tokens.to(device)
            masked_inputs, labels = mask_inputs(test_tokens, vocab_size, mask_id, pad_id)
            logits = model(masked_inputs)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            test_loss += loss.item()

            preds = logits.argmax(dim=-1)
            mask = labels != -100
            total_correct += ((preds == labels) & mask).sum().item()
            total_masked += mask.sum().item()

    avg_test_loss = test_loss / len(test_loader)
    token_accuracy = 100 * total_correct / total_masked if total_masked != 0 else 0.0

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Token Accuracy: {token_accuracy:.2f}%")

if __name__ == '__main__':
    main()
