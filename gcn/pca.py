from sklearn.decomposition import PCA,FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
ff = open('/home/lab106/bjy/PCA/lda_test.txt','w+')
digits_train = np.genfromtxt("{}{}.txt".format('/home/lab106/bjy/PCA/', 'test'),dtype=np.dtype(int))
print (digits_train.shape)
X_digits = digits_train[np.arange(2708)]
estimator = LinearDiscriminantAnalysis(n_components=50)
X_pca=estimator.fit_transform(X_digits)
print(len(X_pca))
for i in range(len(X_pca)):
  string_l = ''
  for num in X_pca[i]:
      string_l = string_l + str(num) + '\t'
  ff.write(string_l)
  ff.write('\n')
ff.close()
print(X_pca.shape)
