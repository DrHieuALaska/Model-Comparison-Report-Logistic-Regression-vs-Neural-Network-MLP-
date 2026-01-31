import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
df = pd.read_csv(DATA_PATH)

# Separate features and labels
X = df.drop(columns=['id', 'diagnosis']).values # Features
y = df['diagnosis'].map({'M': 1, 'B': 0}).values  # Target: Map 'M' to 1 and 'B' to 0

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Split the dataset into training+validation and test sets
from sklearn.model_selection import train_test_split

# train (60%), val (20%), test (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.25,   # 0.25 × 0.8 = 0.2
    random_state=42,
    stratify=y_trainval
)

print(X_train.shape, X_val.shape, X_test.shape)

from sklearn.preprocessing  import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit on train & scale train
X_val_scaled = scaler.transform(X_val)         # scale val
X_test_scaled = scaler.transform(X_test)       # scale test

print("Mean of scaled training data (first 5 features):", X_train_scaled.mean(axis=0)[:5])
print("Std of scaled training data (first 5 features):", X_train_scaled.std(axis=0)[:5])

import joblib
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
print("Scaler saved to scaler.pkl")
print("*********************************************************************")