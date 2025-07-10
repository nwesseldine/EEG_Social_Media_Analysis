import os
from sklearn import tree
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matplotlib import pyplot as plt
import datetime

# Purpose: Training only
# What it does:
    # Loads the dataset
    # Splits data into train/test
    # Trains both RandomForestClassifier and RandomForestRegressor
    # Evaluates models on test set
    # Saves the trained models with timestamps
    # Prints model performance and save locations

data = pd.read_csv("../featuresets/original_data_2025-07-09_15-30.csv")

X_train, X_test, y_train, y_test = train_test_split(data.drop(axis=1, labels=["Label"]), data["Label"], test_size=0.2, random_state=42)

clf = RandomForestClassifier(criterion='entropy', min_samples_leaf=5, max_depth=5)
clf = clf.fit(X_train, y_train)

reg = RandomForestRegressor(criterion='absolute_error', min_samples_leaf=5, max_depth=5)
reg = reg.fit(X_train, y_train)

clf_predictions = clf.predict(X_test)
reg_predictions = reg.predict(X_test)

clf_score = clf.score(X_test, y_test)
reg_score = reg.score(X_test, y_test)

print(f"Classification accuracy: {clf_score:.4f}")
print(f"Regression R² score: {reg_score:.4f}")

# Save the trained model
today = datetime.datetime.now()
datetime_str = today.strftime("%Y-%m-%d_%H-%M")
os.makedirs('../models', exist_ok=True)
joblib.dump(reg, f'../models/concentration_rf_reg_model_{datetime_str}.pkl')
joblib.dump(clf, f'../models/concentration_rf_clf_model_{datetime_str}.pkl')

print(f"Models saved:")
print(f"  - ../models/concentration_rf_reg_model_{datetime_str}.pkl")
print(f"  - ../models/concentration_rf_clf_model_{datetime_str}.pkl")