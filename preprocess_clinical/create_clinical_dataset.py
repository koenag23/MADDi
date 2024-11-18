#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,MaxPooling1D, Flatten,BatchNormalization, GaussianNoise,Conv1D
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import compute_class_weight
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, save_model, load_model


# In[36]:


#this was created in general/diagnosis_making notebook
diag = pd.read_csv("../general/ground_truth.csv").drop("Unnamed: 0", axis=1)


# Below we are combining several clinical datasets.

# In[37]:


demo = pd.read_csv("PTDEMOG.csv").rename(columns={"PHASE":"Phase"})


# In[38]:


neuro = pd.read_csv("NEUROEXM.csv").rename(columns={"PHASE":"Phase"})


# In[39]:


neuro.columns


# In[40]:


clinical = pd.read_csv("ADSP_PHC_COGN.csv").rename(columns={"PHASE":"Phase"})


# In[41]:


clinical.head()


# In[42]:


diag["Subject"].value_counts()


# In[43]:


comb = pd.read_csv("ADSP_PHC_COGN.csv").rename(columns={"PHASE":"Phase"})
comb["PTID"]= comb["SUBJECT_KEY"].str.replace("ADNI_", "").str.replace("s", "S")
comb = comb[["RID", "PTID" , "Phase"]]


# In[44]:


m = comb.merge(demo, on = ["RID", "Phase"]).merge(neuro,on = ["RID", "Phase"]).merge(clinical,on = ["RID", "Phase"]).drop_duplicates()


# In[45]:


m.columns = [c[:-2] if str(c).endswith(('_x','_y')) else c for c in m.columns]

m = m.loc[:,~m.columns.duplicated()]


# In[46]:


diag = diag.rename(columns = {"Subject": "PTID"})


# In[47]:


m = m.merge(diag, on = ["PTID", "Phase"])


# In[48]:


m["PTID"].value_counts()


# In[49]:


t = m
pd.set_option('display.max_columns', None)
print(t)


# In[50]:


t = t.drop(["ID",  "SITEID", "VISCODE", "VISCODE2", "USERDATE", "USERDATE2",
            "update_stamp",  "PTSOURCE", "PTDOB","DX"], axis=1) 


# In[51]:


t.columns


# In[52]:


t = t.fillna(-4)
t = t.replace("-4", -4)
cols_to_delete = t.columns[(t == -4).sum()/len(t) > .70]
t.drop(cols_to_delete, axis = 1, inplace = True)


# In[53]:


len(t.columns)
t.columns


# In[54]:


# t["PTWORK"] = t["PTWORK"].str.lower().str.replace("housewife", "homemaker").str.replace("rn", "nurse").str.replace("bookeeper","bookkeeper").str.replace("cpa", "accounting")


# In[55]:


# t["PTWORK"] = t["PTWORK"].fillna("-4").astype(str)


# In[56]:


# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*teach.*$)', 'education')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*bookkeep.*$)', 'bookkeeper')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*wife.*$)', 'homemaker')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*educat.*$)', 'education')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*engineer.*$)', 'engineer')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*eingineering.*$)', 'engineer') 
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*computer programmer.*$)', 'engineer') 
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*nurs.*$)', 'nurse')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*manage.*$)', 'managment')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*therapist.*$)', 'therapist')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*sales.*$)', 'sales')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*admin.*$)', 'admin')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*account.*$)', 'accounting')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*real.*$)', 'real estate')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*secretary.*$)', 'secretary')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*professor.*$)', 'professor')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*chem.*$)', 'chemist')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*business.*$)', 'business')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*writ.*$)', 'writing')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*psych.*$)', 'psychology')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*analys.*$)', 'analyst')


# In[57]:


# cond = t['PTWORK'].value_counts()
# threshold = 10
# t['PTWORK'] = np.where(t['PTWORK'].isin(cond.index[cond >= threshold ]), t['PTWORK'], 'other')


# In[58]:


# categorical = ['PTGENDER', 'PTWORK',
#  'PTHOME',
#  'PTMARRY',
#  'PTEDUCAT',
#  'PTPLANG',
#  'NXVISUAL',
#  'PTNOTRT',
#  'NXTREMOR',
#  'NXAUDITO',
#  'PTHAND']

# REMOVE PTWORK FOR NOW
categorical = ['PTGENDER',
 'PTHOME',
 'PTMARRY',
 'PTEDUCAT',
 'PTPLANG',
 'NXVISUAL',
 'PTNOTRT',
 'NXTREMOR',
 'NXAUDITO',
 'PTHAND']


# In[59]:


quant = ['PTDOBYY',
 'PHC_MEM',
 'PHC_EXF',
 'PTRACCAT',
 'AGE',
 'PTADDX',
 'PTETHCAT',
 'PTCOGBEG',
 'PHC_VSP',
 'PHC_LAN']


# In[60]:


# text = ["PTWORK", "CMMED"]

# REMOVE PTWORK FOR NOW
text = ["CMMED"]


# In[61]:


cols_left = list(set(t.columns) - set(categorical) - set(text)  - set(["label", "Group","GROUP", "Phase", "RID", "PTID"]))
t[cols_left]


# In[62]:


for col in cols_left:
    if len(t[col].value_counts()) < 10:
        print(col)
        categorical.append(col)


# In[63]:


t.columns


# In[64]:


# to_del = ["PTRTYR", "EXAMDATE", "SUBJECT_KEY", "PTWRECNT"]

#Remove PTWRECNT for now!! -> not in any .csv file
to_del = ["PTRTYR", "EXAMDATE", "SUBJECT_KEY"]
t = t.drop(to_del, axis=1)


# In[65]:


quant = list(set(cols_left) - set(categorical) - set(text)  -set(to_del) - set(["label", "Group","GROUP", "Phase", "RID", "PTID"]))
t[quant]


# In[66]:


cols_left = list(set(cols_left) - set(categorical) - set(text) - set(quant) - set(to_del))


# In[67]:


#after reviewing the meaning of each column, these are the final ones
# l = ['RID', 'PTID', 'Group', 'Phase', 'PTGENDER', 'PTDOBYY', 'PTHAND',
#        'PTMARRY', 'PTEDUCAT', 'PTWORK', 'PTNOTRT', 'PTHOME', 'PTTLANG',
#        'PTPLANG', 'PTCOGBEG', 'PTETHCAT', 'PTRACCAT', 'NXVISUAL',
#        'NXAUDITO', 'NXTREMOR', 'NXCONSCI', 'NXNERVE', 'NXMOTOR', 'NXFINGER',
#        'NXHEEL', 'NXSENSOR', 'NXTENDON', 'NXPLANTA', 'NXGAIT', 
#        'NXABNORM',  'PHC_MEM', 'PHC_EXF', 'PHC_LAN', 'PHC_VSP']

# REMOVE PTWORK FOR NOW
l = ['RID', 'PTID', 'GroupN', 'Phase', 'PTGENDER', 'PTDOBYY', 'PTHAND',
       'PTMARRY', 'PTEDUCAT', 'PTNOTRT', 'PTHOME', 'PTTLANG',
       'PTPLANG', 'PTCOGBEG', 'PTETHCAT', 'PTRACCAT', 'NXVISUAL',
       'NXAUDITO', 'NXTREMOR', 'NXCONSCI', 'NXNERVE', 'NXMOTOR', 'NXFINGER',
       'NXHEEL', 'NXSENSOR', 'NXTENDON', 'NXPLANTA', 'NXGAIT', 
       'NXABNORM',  'PHC_MEM', 'PHC_EXF', 'PHC_LAN', 'PHC_VSP']


# In[68]:


t[l]


# In[69]:


dfs = []


# In[70]:


for col in categorical:
    dfs.append(pd.get_dummies(t[col], prefix = col))


# In[71]:


cat = pd.concat(dfs, axis=1)


# In[72]:


t[quant]


# In[73]:


cat


# In[74]:


t[["PTID","RID", "Phase", "GroupN"]]


# In[75]:


c = pd.concat([t[["PTID", "RID", "Phase", "GroupN"]].reset_index(), cat.reset_index(), t[quant].reset_index()], axis=1).drop("index", axis=1) #tex


# In[76]:


c


# In[77]:


#removing repeating subjects, taking the most recent diagnosis
c = c.groupby('PTID', 
                  group_keys=False).apply(lambda x: x.loc[x["GroupN"].astype(int).idxmax()]).drop("PTID", axis = 1).reset_index(inplace=False)


# In[78]:


c.to_csv("clinical.csv")


# In[79]:


#reading in the overlap test set
ts = pd.read_csv("overlap_test_set.csv").rename(columns={"subject": "PTID"})

#removing ids from the overlap test set
c = c[~c["PTID"].isin(list(ts["PTID"].values))]


# In[ ]:


cols = list(set(c.columns) - set(["PTID","RID","subject", "ID","GROUP", "GroupN", "label", "Phase", "SITEID", "VISCODE", "VISCODE2", "USERDATE", "USERDATE2", "update_stamp", "DX_x","DX_y", "Unnamed: 0"]))
X = c[cols].values 
y = c["GroupN"].astype(int).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[84]:


with open("X_train_c.pkl", 'wb') as X_tr:
    np.save(X_tr, X_train)
    
with open("Y_train_c.pkl", 'wb') as Y_tr:
    np.save(Y_tr, y_train)
    
with open("X_test_c.pkl", 'wb') as X_te:
    np.save(X_te, X_test)
    
with open("Y_test_c.pkl", 'wb') as Y_te:
    np.save(Y_te, y_test)

