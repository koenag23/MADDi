# === ZERO SHOT ATTEMPT ===

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

# === Generate Alzheimer's disease diagnosis inference ===
with torch.no_grad():
    outputs = model.generate(**tabular_input, max_length=512, num_return_sequences=1)

# Decode the model's output
diagnosis_inference = tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Print the inference ===
print("Alzheimer's Disease Diagnosis Inference:")
print(diagnosis_inference)
