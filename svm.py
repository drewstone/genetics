import numpy as np
from sklearn import svm

def fit(tX, tY):
	clf = svm.SVC(gamma='scale')
	clf.fit(tX, tY)
	return clf

def predict(clf, x):
	clf.predict(x)