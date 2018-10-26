import numpy as np
import sklearn
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('DATASET.csv')
df = df.sample(frac=0.0009, random_state = 1)
df.replace('?',-99999, inplace=True)
#df.drop(['ID'], 1, inplace=True)

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = preprocessing.normalize(X_train, norm='l2')

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
predictions=clf.predict(X_test)

x=sklearn.metrics.confusion_matrix(y_test, predictions)
y=sklearn.metrics.accuracy_score(y_test,predictions)
print("Confusion Matrix\n",x)
print("\nTrue Positives : ",x[0][0],"\nFalse Positives:",x[0][1],"\nTrue Negatives :",x[1][0],"\nFalse Negatives:",x[1][1],"\n")
print("\nAccuracy Score/ Efficiency:",y,"\n")
#Precision tells how close we are to the true value=TP/(TP+FP)
print("Precision :",x[0][1]/(x[0][0]+x[0][1]), "\n")
print('ACCURACY SCORE -',  accuracy)


