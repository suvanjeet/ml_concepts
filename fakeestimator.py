import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.dummy import DummyRegressor
X,y= make_regression()
#from sklearn import Dummy
fakeestimator = DummyRegressor()    # stratergy= 'median' will through median values in print row
fakeestimator.fit(X, y)
print (fakeestimator.predict(X)[:5])  # returns mean values all are same 

