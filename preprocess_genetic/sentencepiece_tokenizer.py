import pandas as pd
import sentencepiece as spm

# Load the dataset
file_path = "small_sample.csv"  # Update with your file path
df = pd.read_csv(file_path, engine='python')

# Identify column indices
chromosome_col = df.iloc[:, 0]  # First column (Chromosome)
patient_columns = df.columns[1:]

# # Need to adjust the following ranges based on the number of metadata columns
# patient_columns = df.columns[1:-5]  # Exclude last 5 metadata columns (adjust as needed)
# metadata_columns = df.columns[-5:]  # Last 5 columns are metadata (adjust as needed)

# # When we have the entire dataset (WITH METADATA)
with open("genetic_corpus_patient_view.txt", "w") as f:
    for patient in patient_columns:
        f.write(f"Subject: {patient}\n")  # Patient header
        for idx, row in df.iterrows():
                chromosome = row.iloc[0]  # Chromosome info
                genotype_info = row[patient]  # Genetic data for this patient
                # metadata_info = " | ".join(row[metadata_columns].astype(str))  # Metadata for this gene area
                
                # Formatting output
                gene_info = f"Gene Area: (Chromosome {chromosome}) - {genotype_info}"
                # gene_info = f"Gene Area: {metadata_info} (Chromosome {chromosome}) - {genotype_info}"
                f.write(gene_info + "\n")  # Write to file
        
        f.write("\n")  # Add space between subjects

# Train SentencePiece tokenizer
spm.SentencePieceTrainer.train(input="genetic_corpus_patient_view.txt",
                               model_prefix="genetic_tokenizer_patient_view",
                               vocab_size=8000,  # Adjust based on dataset size
                               character_coverage=0.9995,  # Ensure rare tokens are captured
                               model_type="bpe")  # Can try "unigram", "char", "bpe"

# Load and test tokenizer
sp = spm.SentencePieceProcessor(model_file="genetic_tokenizer_patient_view.model")
print(sp.encode("Gene Area: BRCA1 (Chromosome 1) - Homozygous", out_type=str))  # Tokenized output

