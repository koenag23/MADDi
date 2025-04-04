import pandas as pd
import pickle as pkl
import numpy as np

with open('mri_meta.pkl', 'rb') as i:
    imgs = pkl.load(i)
    
clinical = pd.read_csv('clinical.csv')

clinical_prompts = pd.read_json('clinical_prompts.json')

indexes = list(clinical_prompts.columns)
prompts = [clinical_prompts[index]['question'] for index in clinical_prompts]
clinical_prompts = pd.DataFrame()
clinical_prompts['PTID'] = indexes
clinical_prompts['prompts'] = prompts


imgs = imgs.rename(columns={'subject': 'PTID'})

meta = pd.merge(clinical, imgs, on='PTID', how='outer')
meta = pd.merge(meta, clinical_prompts, on='PTID', how='outer')
meta.drop('GroupN', axis=1, inplace=True)

meta['clinical_exists'] = [True if pd.notnull(rid) else False for rid in meta['RID']]
meta['img_exists'] = [True if pd.notnull(img) else False for img  in meta['coronal']]

meta.to_csv('test.csv')