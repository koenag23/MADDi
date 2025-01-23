# Load model directly
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mlfoundations/tabula-8b")
model = AutoModelForCausalLM.from_pretrained("mlfoundations/tabula-8b")

# === Setting up device ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") <- if we have a GPU
device = torch.device("cpu")
model = model.to(device)

file_path = '../preprocess_clinical/clinical.csv' # INSERT CSV FILE
df = pd.read_csv(file_path)
# Need to convert csv data to serialized format for tabula-8b
tabular_data = "\n".join(df.apply(lambda row: " | ".join(f"{col}: {row[col]}" for col in df.columns), axis=1))
tabular_input = tokenizer(tabular_data, return_tensors="pt")

# REPEAT FOR OTHER TYPES OF TABULAR INPUT (with diff variables)

# Combine inputs (concatenation, cross-attention?)
# Note: may need to fine tune for cross-attention
# NEED TO HAVE TRAINING, TESTING, VALIDATION