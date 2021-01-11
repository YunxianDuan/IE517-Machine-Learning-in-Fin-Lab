from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

Treasury = pd.read_csv("F:/MSFE/machine learning in Fin lab/HW2/Treasury Squeeze test - DS1(1).csv" )
# Create feature and target arrays
X = Treasury.iloc[:, 2:11]
y = Treasury.iloc[:, 11]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Standardizing the features:
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

## Building a decision tree
tree = DecisionTreeClassifier(criterion='gini',
                              max_depth=4,
                              random_state=1)
tree.fit(X_train_std, y_train)

y_pred = tree.predict(X_test_std)

print(accuracy_score(y_test, y_pred))
print("My name is Yunxian Duan")
print("My NetID is: yunxian2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")