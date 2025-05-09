import pandas as pd
import gc
from sklearn.feature_selection import SelectKBest, f_classif

def simplify(string):
    first = string[0]
    second = string[2]
    if not first.isnumeric() or not second.isnumeric():
        return 3
    if first == second == '0':
        return 0
    return 1 if first != second else 2

df = pd.read_csv('combined.csv')
diag = pd.read_csv('diagnosis_full.csv')
latest_diag = diag.drop_duplicates(subset=['Subject'], keep='last')

metadata_cols = [df.columns[0]] + list(df.columns)[-8:]
meta_df = df[metadata_cols]
df_full = df.T
df_data = df.drop(metadata_cols, axis='columns').T
del df, meta_df
gc.collect()

df_data['Subject'] = df_data.index
df_data['Subject'] = df_data['Subject'].apply(lambda x: x.upper())
df_fulldata = df_data.merge(latest_diag[['Subject','GroupN']], how='left', on='Subject')
del df_data
gc.collect()

y = df_fulldata['GroupN']
y.index = df_fulldata['Subject']

df_rawdata = df_fulldata.drop(['Subject','GroupN'], axis='columns')
del df_fulldata
gc.collect()

df_simple = df_rawdata.map(simplify)
del df_rawdata
gc.collect()

X = df_simple.astype('int8')
del df_simple
gc.collect()

# Feature Selection
num_features = 50000
selector = SelectKBest(score_func=f_classif, k=num_features)
X_selected = selector.fit_transform(X, y)
mask = selector.get_support()
selected_snp_names = X.columns[mask]

del X
gc.collect()

best_features = df_full[selected_snp_names]
best_features = best_features.T
best_features.sort_values(by=['POS'])
best_features.to_csv(f'sample_{num_features}.csv')
best_metadata = best_features[metadata_cols]
best_metadata.to_csv('best_feature_meta.csv')
print(best_features.columns)

