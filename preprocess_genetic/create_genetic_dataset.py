import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten,BatchNormalization, GaussianNoise
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import compute_class_weight
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Reading all the SNP files
vcf = pd.read_pickle("all_vcfs.pkl")

# Reading in the diagnosis data
m = pd.read_csv("diagnosis_full.csv").drop("index", axis=1).rename(columns={"Subject": "subject", "GROUP": "label"})

# Making sure all the diagnosis labels are valid
m = m[m["label"] != -1]

# Merging SNPs with diagnosis
vcf = vcf.merge(m[["subject", "label"]], on="subject")
vcf = vcf.drop_duplicates()

# Feature selection: Exclude non-genotype columns
cols = list(set(vcf.columns) - set(["subject", "GROUP", "label"]))

# Preparing the data
X = vcf[cols].values.astype(int)  # Convert genotype values to integers
y = vcf["label"].astype(int).values

# Adjusting labels (if necessary, e.g., to start from 0)
y = y - 1  # Assuming labels start from 1, this maps them to [0, 1, ...]

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# Feature selection using Random Forest on the training data
sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
sel.fit(X_train, y_train)

# Identify selected features
selected_mask = sel.get_support()  # Boolean mask of selected features
selected_features = [cols[i] for i in range(len(cols)) if selected_mask[i]]  # Map back to column names

# Debugging information
print(f"Number of selected features: {len(selected_features)}")
print(f"Selected features: {selected_features}")

# Preparing data with only selected features
X_train_selected = sel.transform(X_train)
X_test_selected = sel.transform(X_test)

# Save the selected features and datasets
vcf_selected = vcf[["label", "subject"] + selected_features]  # Subset with selected features
vcf_selected.to_pickle("vcf_select.pkl")

pd.to_pickle(X_train_selected, "X_train_vcf.pkl")
pd.to_pickle(y_train, "y_train_vcf.pkl")
pd.to_pickle(X_test_selected, "X_test_vcf.pkl")
pd.to_pickle(y_test, "y_test_vcf.pkl")
