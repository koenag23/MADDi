import numpy as np
import skimage.transform as skTrans
import nibabel as nib
import pandas as pd
import os
import shutil
import sys
import zipfile
import tqdm
from PIL import Image as img
import gc

#python3 preprocess_images.py metadir datazips

def find_data_name(filename):
    return filename[:filename.find('T')+1]

def match_meta_to_zip(path_to_datadir_zip, path_to_metadir):
    matching = dict()
    csvs = os.listdir(path_to_metadir)
        
    zips = os.listdir(path_to_datadir_zip)

    for csv in csvs:
        key = path_to_metadir + '/' + csv
        matching[key] = []
        
        mod_csv = csv.replace(' ', '_')
        data_name = find_data_name(mod_csv)
        for zip in zips:
            mod_zip = zip.replace(' ', '_')
            if mod_zip.find(data_name) == -1:
                continue
            value = path_to_datadir_zip + '/' + zip
            matching[key].append(value)

    return matching

def extract_imgs(path_to_datadir):
    
    imgs_path = find_data_name(path_to_datadir) + '/'
    root_dir = 'ADNI/'
    
    os.mkdir(imgs_path)
    with zipfile.ZipFile(path_to_datadir, 'r') as zip_ref:
        zip_ref.extractall(imgs_path)
    
    for root, _, files in os.walk(imgs_path):
        for file in files:
            if file.endswith(".nii"):
                file_path = os.path.join(root, file)
                os.rename(file_path, imgs_path + file)
    
    shutil.rmtree(imgs_path + root_dir)
    print(imgs_path, "extracted")
    return imgs_path

def create_dataset(meta, meta_all, path_to_datadir):
    files = os.listdir(path_to_datadir)
    start = '_'
    end = '.nii'
    
    for file in files:
        if file != '.DS_Store':
            path = os.path.join(path_to_datadir, file)
            img_id = file.split(start)[-1].split(end)[0]
            idx = meta[meta["Image Data ID"] == img_id].index[0]
            im = nib.load(path).get_fdata()
            n_i, n_j, n_k = im.shape
            center_i = (n_i - 1) // 2  
            center_j = (n_j - 1) // 2
            center_k = (n_k - 1) // 2

            im1 = img.fromarray(skTrans.resize(im[center_i, :, :], (72,72), order=1, preserve_range=True))
            im2 = img.fromarray(skTrans.resize(im[:, center_j, :], (72,72), order=1, preserve_range=True))
            im3 = img.fromarray(skTrans.resize(im[:, :, center_k], (72,72), order=1, preserve_range=True))
            
            label = meta.at[idx, "Group"]
            subject = meta.at[idx, "Subject"]
            visit = meta.at[idx, "Visit"]

            frame = np.array([im1, im2, im3, label, subject, visit], dtype=object)
            meta_all.index += 1
            meta_all.loc[0] = frame

def main():
    args = sys.argv[1:]
    path_to_metadir = args[0] 
    path_to_datadir_zip = args[1]
    
    matching = match_meta_to_zip(path_to_datadir_zip, path_to_metadir)
    
    meta_all = pd.DataFrame(columns = ["im1","im2","im3","label","subject","visit"], dtype=object)
    
    for metacsv in matching.keys():
        meta = pd.read_csv(metacsv)
        print("Opened", metacsv)
        #get rid of not needed columns
        meta = meta[["Image Data ID", "Group", "Subject", "Visit"]] #MCI = 0, CN =1, AD = 2
        meta["Group"] = pd.factorize(meta["Group"])[0]
        
        datazips = matching[metacsv]
        datazips.sort()
        for datazip in tqdm.tqdm(datazips):
            path_to_datadir = extract_imgs(datazip)
            
            #initialize new dataset where arrays will go
            create_dataset(meta, meta_all, path_to_datadir)
            shutil.rmtree(path_to_datadir)
            
            gc.collect()
    
    meta_all = meta_all.sort_index()
    meta_all.to_pickle("mri_meta.pkl")
            

if __name__ == '__main__':
    
    main()
    
