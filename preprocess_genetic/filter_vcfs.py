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
    # Example:
    # files = os.listdir("vcf_dir")
    files = os.listdir("YOUR_PATH_TO_VCFS")
    
    for vcf_file in files:
        # Example:
        # vcf_file = "ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr21.vcf.gz"
        # file_name = "vcf_dir/ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr21.vcf.gz"
        file_name = "YOUR_PATH_TO_VCFS" + vcf_file
        
        output_file = open('log.txt','a')
        output_file.write(file_name)
        output_file.close()
        names = get_vcf_names(file_name)
        vcf_chunks = pd.read_csv(file_name, compression='gzip', comment='#', chunksize=10000, delim_whitespace=True, header=None, names=names)
 
        # To process each chunk at a time and add it to the data frame
        # Note: currently set to only process first 11 chunks
        # vcf = pd.DataFrame()
        # for i, chunk in enumerate(vcf_chunks):
        #     print(f"Processing chunk {i + 1}, shape: {chunk.shape}")
        #     vcf = pd.concat([vcf, chunk], ignore_index=True)

        #     if (i == 10):
        #         break
        # print("Final DataFrame shape:", vcf.shape)

        vcf = pd.concat(vcf_chunks, ignore_index=True)

        
        start = vcf_file.find("ADNI_ID.") + len("ADNI_ID.")
        end = vcf_file.find(".vcf")
        substring = vcf_file[start:end]
        relevent = genes[genes["chrom"] == substring]
        relevent = relevent.reset_index()
        positions = vcf["POS"]
        
        
        indexes = []
        for i in range(len(positions)):
            boo = in_between(positions[i], relevent)
            if i % 500 == 0:
                output_file = open('log.txt','a')
                output_file.write(" " + str(boo) + " ")
                output_file.close()
            if boo:
                indexes.append(i)

        
        if len(indexes) != 0:
            df = vcf.iloc[indexes]
            df.to_pickle(vcf_file[:-4] + ".pkl")
        
    

    
if __name__ == '__main__':
    main()
    