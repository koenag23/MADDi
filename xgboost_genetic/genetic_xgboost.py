import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle


#loading the data
def load_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


X_train = load_pickle("X_train_vcf.pkl")
X_test = load_pickle("X_test_vcf.pkl")
y_train = load_pickle("y_train_vcf.pkl")
y_test = load_pickle("y_test_vcf.pkl")


# Convert labels to 0, 1, 2 from -1, 0, 1
y_train = y_train + 1
y_test = y_test + 1


#Train the model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
num_classes = len(np.unique(y_train))
params = {
    'objective': 'multi:softmax',  # For multi-class classification with discrete outputs
    'num_class': num_classes,
    'eval_metric': 'mlogloss',    # Multi-class logarithmic loss
    'max_depth': 6,               # Maximum tree depth
    'eta': 0.1,                   # Learning rate
    'seed': 42                    # Seed for reproducibility
}
bst = xgb.train(params, dtrain, num_boost_round=100)

#Evaluate the model
# Predict class labels (output will be 0, 1, or 2 directly)
y_pred = bst.predict(dtest)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Alzheimer\'s', 'Mild', 'Severe']))