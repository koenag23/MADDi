import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 1. Load the data
file_path = './preprocess_clinical/clinical.csv'
#file_path = '/absolute/path/to/clinical.csv'
data = pd.read_csv(file_path)


list_of_keys = """PTGENDER_-4.0,
PTGENDER_1.0,
PTGENDER_2.0,
PTHOME_-4.0,
PTHOME_1.0,
PTHOME_2.0,
PTHOME_3.0,
PTHOME_4.0,
PTHOME_5.0,
PTHOME_6.0,
PTHOME_8.0,
PTMARRY_-4.0,
PTMARRY_1.0,
PTMARRY_2.0,
PTMARRY_3.0,
PTMARRY_4.0,
PTMARRY_5.0,
PTEDUCAT_-4.0,
PTEDUCAT_6.0,
PTEDUCAT_7.0,
PTEDUCAT_8.0,
PTEDUCAT_9.0,
PTEDUCAT_10.0,
PTEDUCAT_11.0,
PTEDUCAT_12.0,
PTEDUCAT_13.0,
PTEDUCAT_14.0,
PTEDUCAT_15.0,
PTEDUCAT_16.0,
PTEDUCAT_17.0,
PTEDUCAT_18.0,
PTEDUCAT_19.0,
PTEDUCAT_20.0,
PTPLANG_-4.0,
PTPLANG_1.0,
PTPLANG_2.0,
PTPLANG_3.0,
NXVISUAL_1.0,
NXVISUAL_2.0,
PTNOTRT_-4.0,
PTNOTRT_0.0,
PTNOTRT_1.0,
PTNOTRT_2.0,
NXTREMOR_1.0,
NXTREMOR_2.0,
NXAUDITO_1.0,
NXAUDITO_2.0,
PTHAND_-4.0,
PTHAND_1.0,
PTHAND_2.0,
NXMOTOR_1.0,
NXMOTOR_2.0,
NXSENSOR_1.0,
NXSENSOR_2.0,
PTTLANG_-4.0,
PTTLANG_1.0,
PTTLANG_2.0,
NXCONSCI_1.0,
NXCONSCI_2.0,
NXGAIT_-4.0,
NXGAIT_1.0,
NXGAIT_2.0,
PTETHCAT_-4.0,
PTETHCAT_1.0,
PTETHCAT_2.0,
PTETHCAT_3.0,
NXHEEL_-4.0,
NXHEEL_1.0,
NXHEEL_2.0,
GroupN_0.0,GroupN_1.0,GroupN_2.0,
NXNERVE_1.0,
NXNERVE_2.0,
PTRACCAT_-4,
PTRACCAT_1,
PTRACCAT_2,
PTRACCAT_3,
PTRACCAT_4,
PTRACCAT_5,
PTRACCAT_6,
PTRACCAT_7,
NXTENDON_1.0,
NXTENDON_2.0,
NXOTHER_-4.0,
NXOTHER_-1.0,
NXOTHER_1.0,
NXOTHER_2.0,
NXPLANTA_1.0,
NXPLANTA_2.0,
NXABNORM_-4.0,
NXABNORM_1.0,
NXABNORM_2.0,
NXFINGER_1.0,
NXFINGER_2.0,
PTDOBYY,
PTCOGBEG,
PHC_MEM,
PTADDX,
PHC_LAN,
PHC_EXF,
PHC_VSP"""
list_of_keys = list_of_keys.split(',')

# 2. Preprocess the data
X = data.drop(columns=['GroupN'])  # Features
<<<<<<< Updated upstream
X = X.drop(columns=['PTID', 'RID', 'Phase', 'VISDATE'])
X = X.drop(columns=['GroupN_0.0','GroupN_1.0','GroupN_2.0'])
=======
X = X.drop(columns=['PTID', 'RID', 'Phase', 'VISDATE', 'GroupN_0.0', 'GroupN_1.0', 'GroupN_2.0'])
>>>>>>> Stashed changes
X = X.drop(X.columns[0], axis=1)
print(X.shape)
y = data['GroupN']  # Target

# 3. Split the data
<<<<<<< Updated upstream
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
=======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
>>>>>>> Stashed changes

# 4. Train the XGBoost model
# Convert to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up parameters for multi-class classification
num_classes = len(y.unique())  # Number of classes (in this case, 3: 0, 1, 2)
params = {
    'objective': 'multi:softmax',  # For multi-class classification with discrete outputs
    'num_class': num_classes,     # Number of classes
    'eval_metric': 'mlogloss',    # Multi-class logarithmic loss
    'max_depth': 6,               # Maximum tree depth
    'eta': 0.1,                   # Learning rate
    'seed': 42                    # Seed for reproducibility
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=100)

# 5. Evaluate the model
# Predict class labels (output will be 0, 1, or 2 directly)
y_pred = bst.predict(dtest)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Alzheimer\'s', 'Mild', 'Severe']))