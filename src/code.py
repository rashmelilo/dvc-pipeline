import os
import json
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)

import pickle
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


ete = pd.read_csv("./data/bhutan_landslide_data.csv")
ete["Type"] = ete["Type"].astype("category")
ete = ete.drop(["Type"], axis=1)

features = [
    "Lithology",
    "Altitude",
    "Slope",
    "Total curvature",
    "Aspect",
    "Distance to road",
    "Distance to stream",
    "Slope length",
    "TWI",
    "STI",
]

X = ete.loc[:, features]
y = ete.loc[:, ["Code"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=8, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=8, stratify=y_train
)

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(X_val.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_val shape: {}".format(y_train.shape))
print("y val shape: {}".format(y_val.shape))

y_train = y_train.values.flatten()
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


param_dist = {"n_estimators": randint(50, 500), "max_depth": randint(1, 20)}
rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_
print("Best hyperparameters:", rand_search.best_params_)

y_pred = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

metrics = {
    'precision': precision,
    'accuracy': accuracy,
    'recall': recall,
    'f1_score': f1score
}

json.dump(metrics, open("metrics.json", 'w'))

os.makedirs("models", exist_ok=True)
pickle.dump(rf, open('models/model.pkl', 'wb'))