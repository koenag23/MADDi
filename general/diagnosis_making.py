#!/usr/bin/env python
# coding: utf-8

# ## Combbine all diagnosis

# This method take diagnosis from images, clinical, and diagnosis sheet, and creates one ground truth (where all three agree) and one majority vote (where two agree) diagnosis files. 

# In[1]:


import pandas as pd
import math
clinical = pd.read_csv("ADSP_PHC_COGN.csv").rename(columns={"PHASE":"Phase"})
#this file is the metadata file that one can get from downloading MRI images from ADNI
img = pd.read_csv("metadata.csv")
comb = pd.read_csv("DXSUM_PDXCONV_ADNIALL.csv").rename(columns={"PHASE":"Phase"})
comb = comb[["RID", "PTID" , "Phase"]]


# In[2]:


def read_diagnose(file_path: str = 'DXSUM_PDXCONV_ADNIALL.csv', verbose=False):
    # Read diagnostic summary
    diagnostic_summary = pd.read_csv(file_path, index_col='PTID').rename(columns={"PHASE":"Phase"})
    diagnostic_summary = diagnostic_summary.sort_values(by=["update_stamp"], ascending=True)
    # Create dictionary
    diagnostic_dict: dict = {}
    for key, data in diagnostic_summary.iterrows():
        # Iterate for each row of the document
        phase: str = data['Phase']
        diagnosis: float = -1.
        if phase == "ADNI1":
            diagnosis = data['DIAGNOSIS']
        elif phase == "ADNI2" or phase == "ADNIGO":
            diagnosis = data['DIAGNOSIS']
        elif phase == "ADNI3":
            diagnosis = data['DIAGNOSIS']
        elif phase == "ADNI4":
            diagnosis = data['DIAGNOSIS']
        else:
            print(f"ERROR: Not recognized study phase {phase}")
            exit(1)
        # Update dictionary
        if not math.isnan(diagnosis):
            diagnostic_dict[key] = diagnosis
    if verbose:
        print_diagnostic_dict_summary(diagnostic_dict)
    return diagnostic_dict


def print_diagnostic_dict_summary(diagnostic_dict: dict):
    print(f"Number of diagnosed patients: {len(diagnostic_dict.items())}\n")
    n_NL = 0
    n_MCI = 0
    n_AD = 0
    for (key, data) in diagnostic_dict.items():
        if data == 1:
            n_NL += 1
        if data == 2:
            n_MCI += 1
        if data == 3:
            n_AD += 1
    print(f"Number of NL patients: {n_NL}\n"
          f"Number of MCI patients: {n_MCI}\n"
          f"Number of AD patients: {n_AD}\n")


# In[3]:


d = read_diagnose()
print_diagnostic_dict_summary(d)


# In[4]:


new = pd.DataFrame.from_dict(d, orient='index').reset_index()
print(new)


# In[5]:


clinical.head()


# In[6]:


clinical["year"] = clinical["EXAMDATE"].str[:4]


# In[7]:


clinical["Subject"] = clinical["SUBJECT_KEY"].str.replace("ADNI_", "").str.replace("s", "S")


# In[8]:


c = comb.merge(clinical, on = ["RID", "Phase"])


# In[9]:


c = c.drop("Subject", axis =1)


# In[10]:


c = c.rename(columns = {"PTID":"Subject"})


# In[11]:


img["year"] = img["EXAMDATE"].str[5:].str.replace("/", "")


# In[12]:


img = img.replace(["CN", "MCI", "AD"], [ 0, 1, 2])


# In[13]:


c["DX"] = c["DX"] -1


# In[14]:


new[0] = new[0].astype(int) -1
print(new)


# In[15]:


new = new.rename(columns = {"index":"Subject", 0:"GroupN"})
print(new)


# In[16]:


img = img.rename(columns = {"PTID":"Subject", "RECNO":"Group"})


# In[17]:


m = new.merge(c, on = "Subject", how = "outer")
print(m)


# In[18]:


m[["GroupN", "DX"]]


# In[19]:


m = m[["Subject", "GroupN", "DX", "Phase"]].drop_duplicates()


# In[20]:


m = m.dropna(subset = ["GroupN", "DX"], how="all").drop_duplicates()
m


# In[22]:


m


# In[24]:


m3 = m[m["GroupN"] == m["DX"]]


# In[ ]:





# In[ ]:





# In[26]:


m3 = m3[["Subject", "GroupN", "DX", "Phase"]]
m3


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


m5 = m3
i = m5


# In[28]:


i = i.drop_duplicates()


# In[29]:


i


# In[30]:


i[["Subject", "GroupN", "Phase"]].to_csv("ground_truth.csv")


# In[ ]:


m.update(m5[~m5.index.duplicated(keep='first')])


# In[38]:


indexes = m.index


# In[35]:


#if none of the three diagnosis agree, then we set the value to -1
print(m)
m["GROUP"] = -1


# In[ ]:


for i in indexes:
    row = m.loc[i]
    if (row["GroupN"] == row["DX"]):
        val = row["GroupN"]
        m.loc[i, "GROUP"] = val


# In[41]:


m5 = m5[~m5.index.duplicated(keep='first')]
m5


# In[ ]:


m[m["GROUP"] != -1]


# In[44]:


m[["Subject", "GroupN", "DX", "GROUP", "Phase"]].to_csv("diagnosis_full.csv")

