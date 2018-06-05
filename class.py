import pandas as pd
from sklearn import cross_validation, neighbors
import numpy as np

ds = pd.read_csv("breast-cancer-wisconsin.data.txt")
ds.replace('?', -99999, inplace=True)
ds.drop(['id'], 1, inplace=True)

x = np.array(ds.drop(['class'], 1))
y = np.array(ds['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
acuracy = clf.score(x_test, y_test)
print(acuracy)

k = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 6, 5, 5, 2, 3, 2, 8]])
k = k.reshape(len(k), -1)
prediction = clf.predict(k)

print(prediction)


