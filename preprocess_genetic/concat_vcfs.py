import io
import os
import numpy as np
import pandas as pd
import pickle as pkl


def main():
    
    
    
    files = os.listdir("gene_data/")
    diag = pd.read_csv("../general/diagnosis_full.csv")[["Subject", "GroupN"]]
    
    vcfs = []
    
    for vcf_file in files:
        file_name = "gene_data/" + vcf_file
        
        #vcf = pd.read_pickle(file_name)
        
        with open(file_name, "rb") as f:
            object = pkl.load(f)
            
        df = pd.DataFrame(object)
        df.to_csv(r'file.csv')
        
        print(vcf["Subject"])
	
        vcf = vcf.drop(['#CHROM', 'POS', 'ID','REF','ALT','QUAL','FILTER','INFO', 'FORMAT'], axis=1)
        vcf = vcf.T
        vcf.reset_index(level=0, inplace=True)
        vcf["Subject"] = vcf["Subject"].str.replace("s", "S").str.replace("\n", "")
        merged = diag.merge(vcf, on = "Subject")
        merged = merged.rename(columns={"Subject": "subject"})
        d = {'0/0': 0, '0/1': 1, '1/0': 1,  '1/1': 2, "./.": 3}
        cols = list(set(merged.columns) - set(["subject", "GroupN"]))
        for col in cols:
            merged[col] = merged[col].str[:3].replace(d)
            idx = cols.index(col)
            if idx % 500 == 0:
                output_file = open('log_clean.txt','a')
                output_file.write("Percent done: " + str((idx/len(cols))*100) + "\n")
                output_file.close()
        
        merged.to_pickle(vcf_file + "clean.pkl")

        vcf = vcf.groupby('Subject', group_keys=False).apply(lambda x: x.loc[x.Group.idxmax()])

        vcfs.append(vcf)
    
    vcf = pd.concat(vcfs, ignore_index=True)
    vcf = vcf.drop_duplicates()
    vcf.to_pickle("all_vcfs.pkl")



    
if __name__ == '__main__':
    main()
    
