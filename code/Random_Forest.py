from sklearn import tree
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matplotlib import pyplot as plt
import datetime

# Load the dataset

data = pd.read_csv("dataset/output.csv")

X_train, X_test, y_train, y_test = train_test_split(data.drop(axis=1, labels=["Label"]), data["Label"], test_size=0.2, random_state=42)

clf = RandomForestClassifier(criterion='entropy', min_samples_leaf=5, max_depth=5)
clf = clf.fit(X_train, y_train)

reg = RandomForestRegressor(criterion='absolute_error', min_samples_leaf=5, max_depth=5)
reg = reg.fit(X_train, y_train)

clf_predictions = clf.predict(X_test)
reg_predictions = reg.predict(X_test)

clf_score = clf.score(X_test, y_test)
reg_score = reg.score(X_test, y_test)

print(clf_score)
print(reg_score)

real_data = pd.read_csv("featuresets/original_data_2025-07-09_15-30.csv")
real_data = real_data.drop(axis=1, labels=["Label"])

real_clf_predictions = clf.predict(real_data)
real_reg_predictions = reg.predict(real_data)

plt.plot(real_reg_predictions)
plt.title("Predictions on Real Data")
plt.xlabel("Sample Index")
plt.ylabel("Predicted Label")
plt.show()


# Save the trained model
today = datetime.datetime.now()
datetime_str = today.strftime("%Y-%m-%d_%H-%M")
joblib.dump(reg, f'models/concentration_rf_reg_model_{datetime_str}.pkl')

# When you want to load it later
#reg_loaded = joblib.load('random_forest_model.pkl')