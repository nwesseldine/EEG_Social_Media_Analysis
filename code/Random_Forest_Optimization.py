import os
from sklearn import tree
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
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

dataset_name = "Mendeley"
data = pd.read_csv("featuresets/Mendeley cleaned_2025-07-11_13-52.csv", nrows= 1000).drop(labels="Timestep", axis=1)

X_train, X_test, y_train, y_test = train_test_split(data.drop(axis=1, labels=["Label"]), data["Label"], test_size=0.2, random_state=42)
# confusion_matrix(X_test, y_test,  labels = [1, 2, 3])
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2,3,5],
    'min_samples_leaf': [1, 3, 5],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}
#Best Parameters: {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
# grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5, verbose = 4, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# random_search = RandomizedSearchCV(RandomForestClassifier(),
#                                    param_grid, verbose = 4, sn_jobs=-1)
# random_search.fit(X_train, y_train)
# clf = RandomForestClassifier(**grid_search.best_params_, verbose = 4, n_jobs=-1)
# clf = clf.fit(X_train, y_train)

# reg = RandomForestRegressor(criterion='absolute_error', min_samples_leaf=5, max_depth=5, verbose = 4, n_jobs=-1)
# reg = reg.fit(X_train, y_train)

# clf_predictions = clf.predict(X_test)
# reg_predictions = reg.predict(X_test)

# clf_score = clf.score(X_test, y_test)
# reg_score = reg.score(X_test, y_test)




print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)
# print(f"Classification accuracy: {clf_score:.4f}")
# print(f"Regression R² score: {reg_score:.4f}")
# print(classification_report(clf_predictions, y_test))

# print(classification_report(reg_predictions, y_test))
# Save the trained model
# today = datetime.datetime.now()
# datetime_str = today.strftime("%Y-%m-%d_%H-%M")
# joblib.dump(reg, f'models/{dataset_name}_rf_reg_model_{datetime_str}.pkl')
# joblib.dump(clf, f'models/{dataset_name}_rf_clf_model_{datetime_str}.pkl')

# print(f"Models saved:")
# print(f"  - models/Mendeley_rf_reg_model_{datetime_str}.pkl")
# print(f"  - models/Mendeley_rf_clf_model_{datetime_str}.pkl")