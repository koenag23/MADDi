import pickle
import io
import os
import pandas as pd
import gzip
import gc
import numpy as np
from tqdm import tqdm

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
    
    file_name = "vcf_collection/" + vcf_file
        
    
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
    
    unique = set()
    
    for i in range(len(starts)):
        unique.add((starts[i], ends[i]))
        
    unique = sorted(list(unique))
    pos_array = np.asarray(unique)
    
    # Concatenate with debug output
    frames = []
    indexed_items = enumerate(vcf_chunks)
    for num, chunk in indexed_items:
        print(num)
        indexes = []
        positions = chunk["POS"]
        for i, position in enumerate(positions):
            for start, end in pos_array:
                if (position >= start) and (position <= end):
                    indexes.append(i)
                    
            
        if len(indexes) != 0:
            frames.append(chunk.iloc[indexes])
            
    if len(frames) != 0:
        df = pd.concat(frames, ignore_index=True)
        df.to_pickle(vcf_file[:-7] + ".pkl")
        gc.collect()
        

def main():
    
    """ with open("ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr3.pkl", "rb") as file:
        test = pickle.load(file)
        
    test.to_csv("test.csv") """
    
    preprocess_file("ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr18.vcf.gz")
    # folder name of where the vcf files are stored
    """ files = os.listdir("vcf_collection/")
    files = [file for file in files if file.endswith(".gz")]
    files.sort()
    #print(files)
    for file in files[14:]:
        preprocess_file(file)  """
    
    

    
if __name__ == '__main__':
    main()
    
