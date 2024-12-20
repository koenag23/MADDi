
import io
import os
import pandas as pd
import gzip
import gc
import numpy as np

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

def preprocess_file(vcf_file):
    
    # path name of the genes list
    genes = pd.read_csv("gene_list.csv")
    
    file_name = "vcfs/" + vcf_file
        
    print(vcf_file)
    
    names = get_vcf_names(file_name)
    chunk_size = 5000
    vcf_chunks = pd.read_csv(file_name, compression='gzip', comment='#', chunksize=chunk_size, sep=r'\s+', header=None, names=names)
    
    start = vcf_file.find("ADNI_ID.") + len("ADNI_ID.")
    end = vcf_file.find(".vcf")
    substring = vcf_file[start:end]
    relevent = genes[genes["chrom"] == substring]
    relevent = relevent.reset_index()
    
    starts = relevent['start'].to_numpy()
    ends = relevent['end'].to_numpy()
    
    starts = np.sort(starts)
    ends = np.sort(ends)
    
    
    n_interval = len(starts)
    duplicate = (0,0)
    i = 0
    while(i < n_interval):
        if starts[i] == duplicate[0] and ends[i] == duplicate[1]:
            starts = np.delete(starts, i)
            ends = np.delete(ends, i)
            n_interval = len(starts)
        if i == len(starts):
            break
        duplicate = (starts[i], ends[i])
        i += 1
        
            
    n_interval = len(starts)
    
    # Concatenate with debug output
    frames = []
    indexed_items = enumerate(vcf_chunks)
    print(indexed_items)
    for num, chunk in indexed_items:
        indexes = []
        print(f"Processing chunk {num + 1}, shape: {chunk.shape} for {file_name}")
        positions = chunk["POS"]
        for i, position in enumerate(positions):
            for j in range(n_interval):
                start, end = starts[j], ends[j]
                if (position >= start) and (position <= end):
                    indexes.append(i)
                    break
            
        if len(indexes) != 0:
            frames.append(chunk.iloc[indexes])
            
    df = pd.concat(frames, ignore_index=True)
    df.to_pickle(vcf_file[:-7] + ".pkl")
    gc.collect()

def main():
    
    # folder name of where the vcf files are stored
    files = os.listdir("vcfs/")
    files.sort()
    
    preprocess_file(files[1])
    """ for file in files:
        preprocess_file(file) """
    

    
if __name__ == '__main__':
    main()
    