from sklearn.tree import DecisionTreeClassifier
import iris_dataset
import visualize
import pandas as pd

X_y = iris_dataset.get_target_features()

X = X_y[0]  # target
y = X_y[1]  # features

df=pd.read_csv("iris.csv")

clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
visualize.visualize_tree(clf, list(df.columns)[0:4])



