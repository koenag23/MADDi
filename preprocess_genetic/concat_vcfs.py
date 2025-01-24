import io
import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
def assign_genotype_value(genotype):
    """
    Assign values to genotypes based on the following rules:
    - 0/0 -> 0
    - n/n -> 1 (where n is any non-zero digit and both alleles are the same)
    - n/m or m/n -> 2 (where n and m are different non-zero digits)
    - ././ -> 3 (missing data)
    
    Parameters:
        genotype (str): The genotype string in the format "a/b".
    
    Returns:
        int: The assigned value based on the genotype.
    """
    if genotype == "0/0":
        return 0
    elif genotype == "./.":
        return 3
    else:
        try:
            # Split the genotype into two alleles
            alleles = genotype.split("/")
            # Convert alleles to integers for comparison
            allele1, allele2 = int(alleles[0]), int(alleles[1])
            
            if allele1 == allele2 and allele1 != 0:
                return 1  # n/n
            elif allele1 != allele2:
                return 2  # n/m or m/n
        except (ValueError, IndexError):
            # If parsing fails, treat it as missing or invalid data
            return 3

def main():
    
    files = os.listdir("output_directory/")
    diag = pd.read_csv("diagnosis_full.csv")[["Subject", "GroupN"]]
    
    vcfs = []
    
    for vcf_file in tqdm(files):
        file_name = "output_directory/" + vcf_file
        
        vcf = pd.read_pickle(file_name)
        vcf = vcf.iloc[:, 9:]
        vcf = vcf.T
        vcf.reset_index(inplace=True)
        vcf.rename(columns={"index": "Subject"}, inplace=True)
        
        merged = diag.merge(vcf, on = "Subject")
        merged = merged.rename(columns={"Subject": "subject"})
        cols = list(set(merged.columns) - set(["subject", "GroupN"]))
        for col in tqdm(cols, desc=f"Processing columns in {vcf_file}"):
            merged[col] = merged[col].apply(lambda x: assign_genotype_value(str(x)[:3]) if pd.notna(x) else 3) 

        vcfs.append(merged)
    
    vcf = pd.concat(vcfs, ignore_index=True)
    vcf = vcf.drop_duplicates()
    vcf.to_pickle("all_vcfs.pkl")



    
if __name__ == '__main__':
    main()
    
