from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
# Load the dataset

data = pd.read_csv("dataset/output.csv")

X_train, X_test, y_train, y_test = train_test_split(data.drop(axis=1, labels=["Label"]), data["Label"], test_size=0.2, random_state=42)

clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5, max_depth=5)
clf = clf.fit(X_train, y_train)

reg = tree.DecisionTreeRegressor(criterion='absolute_error', min_samples_leaf=5, max_depth=5)
reg = reg.fit(X_train, y_train)

clf_predictions = clf.predict(X_test)
reg_predictions = reg.predict(X_test)

clf_score = clf.score(X_test, y_test)
reg_score = reg.score(X_test, y_test)

print(clf_score)
print(reg_score)