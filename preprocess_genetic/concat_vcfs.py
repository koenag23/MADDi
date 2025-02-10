import io
import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import gc

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
    
    if genotype == "./.":
        return 3
    
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
    
    
def handle_row(row):
    row[2:] = row[2:].apply(assign_genotype_val)
    return row
    
def assign_genotype_val(col):
    if type(col) == int:
        return 3
    genotype = col[:3]
    if genotype == "0/0":
        return 0
    if '0' in genotype:
        return 3
    if not genotype[0].isnumeric() or not genotype[2].isnumeric():
        return 3
    allele1, allele2 = genotype.split("/")
    if allele1 == allele2:
        return 1
    if allele1 != allele2:
        return 2
    return 3
    
def main():
    
    files = os.listdir("output_directory/")
    diag = pd.read_csv("diagnosis_full.csv")[["Subject", "GroupN"]]
    
    matching = {diag['Subject'][i]: int(diag["GroupN"][i]) for i in range(len(diag['Subject']))}

    vcfs = []
    
    for vcf_file in files:
        
        chr_index = vcf_file.find("chr")
        end = vcf_file.rfind(".")
        chromosome = vcf_file[chr_index:end]
        if chromosome == "chr3":
            continue
        
        file_name = "output_directory/" + vcf_file
        
        vcf = pd.read_pickle(file_name)
        
        vcf = vcf.iloc[:, 9:]
        vcf = vcf.T
        vcf.reset_index(inplace=True)
        vcf.rename(columns={"index": "Subject"}, inplace=True)
        
        vcf.to_csv(chromosome + "_data.csv")
        
        vcf_chunks = pd.read_csv(chromosome + "_data.csv", chunksize=50)
        
        
        frames = np.array([])
        for num, chunk in tqdm(enumerate(vcf_chunks)):
            subjects = chunk['Subject'].apply(lambda x: x.upper().strip())
            chunk['GroupN'] = [matching[subj] for subj in subjects]
            new_chunk = chunk.apply(lambda x: handle_row(x), axis=1)
            new_chunk = new_chunk.drop(new_chunk.columns[[0]], axis=1)
            new_chunk.to_csv(chromosome + ".csv")
            break
            #frames = np.append(frames, new_chunk)
        
        continue
        
        print(frames.shape)
        frames = list(frames)
        vcf = pd.concat(frames, ignore_index=True)
        vcf = vcf.drop_duplicates()
        vcf.to_csv("test3.csv")

        """ merged = diag.merge(vcf, on = "Subject")
        merged = merged.rename(columns={"Subject": "subject"})
        merged.replace(to_replace="0/0", value=0) """
        
        
        #cols = list(set(merged.columns) - set(["subject", "GroupN"]))
        """ for col in tqdm(cols, desc=f"Processing columns in {vcf_file}"):
            merged[col] = merged[col].apply(lambda x: assign_genotype_value(str(x)[:3]) if pd.notna(x) else 3, meta=meta_df).compute() """

        """ merged.to_csv('test3-*.csv')
        
        vcfs.append(merged) """
    
    """ vcf = pd.concat(vcfs, ignore_index=True)
    vcf = vcf.drop_duplicates() """
    #vcf.to_pickle("all_vcfs.pkl")



    
if __name__ == '__main__':
    main()
    
