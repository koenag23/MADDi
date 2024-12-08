import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 1. Load the data
file_path = '../preprocess_clinical/clinical.csv'
data = pd.read_csv(file_path)

# 2. Preprocess the data
# Assuming the target column is named 'Diagnosis' (0, 1, or 2)
X = data.drop(columns=['GroupN'])  # Features
X = X.drop(columns=['PTID', 'RID', 'Phase', 'VISDATE'])
X = X.drop(X.columns[0], axis=1)
y = data['GroupN']  # Target

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.value_counts())
print(y_test.value_counts())

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