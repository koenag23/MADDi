
import io
import os
import numpy as np
import pandas as pd
import gzip

def get_vcf_names(vcf_path):
    with gzip.open(vcf_path, "rt") as ifile:
          for line in ifile:
            if line.startswith("#CHROM"):
                vcf_names = [x for x in line.split('\t')]
                break
    ifile.close()
    return vcf_names


def read_vcf(path):
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})

def in_between(position, relevent):
    appears = False
    for i in range(len(relevent)):
        row = relevent.iloc[i]
        if (position >= relevent.iloc[i].start) and (position <= relevent.iloc[i].end):
            appears = True
    return appears

def main():
    
    
    genes = pd.read_csv("gene_list.csv")
    files = os.listdir("vcf_collection/")
    
    
    for vcf_file in files:
        file_name = "vcf_collection/" + vcf_file
        
        print(vcf_file)
        
        names = get_vcf_names(file_name)
        chunk_size = 50000
        vcf_chunks = pd.read_csv(file_name, compression='gzip', comment='#', chunksize=chunk_size, sep=r'\s+', header=None, names=names)
        
        start = vcf_file.find("ADNI_ID.") + len("ADNI_ID.")
        end = vcf_file.find(".vcf")
        substring = vcf_file[start:end]
        relevent = genes[genes["chrom"] == substring]
        relevent = relevent.reset_index()
        
        # Concatenate with debug output
        indexes = []
        frames = []
        for num, chunk in enumerate(vcf_chunks):
            indexes = []
            print(f"Processing chunk {num + 1}, shape: {chunk.shape}")
            
            positions = chunk["POS"]
        
            for i in range(len(positions)):
                index = i + num * chunk_size
                if in_between(positions[index], relevent):
                    indexes.append(i)
            
            if len(indexes) != 0:
                frames.append(chunk.iloc[indexes])
        
        df = pd.concat(frames, ignore_index=True)
        df.to_pickle(vcf_file[:-7] + ".pkl")
        
    

    
if __name__ == '__main__':
    main()
    
