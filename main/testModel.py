import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import pandas as pd
a = np.load('../data/selected_feature.npy')
X_train = pd.read_csv('../data/data_cleaned_train_X.csv')
X_train = X_train.drop(columns=["Unnamed: 0.1"])
  
X_train = X_train.drop(columns=["shared_bath"])
    
X_train = X_train.drop(columns=["private_bath"])
   
print(X_train.columns)
col_set =[]
print(a)
# for i in range(len(coeffs)):
#         if (coeffs[i]):
           
# X_train = X_train[list(col_set)]
# print(type(X_train))
# filename = 'finalized_model.sav'
# loaded_model = joblib.load(filename)
# a =[[0.02777778, 0.13333333, 1.,         0.97 ,      0.23417333, 0.02857143,
#   1.   ,      0. ,       1.   ,      0.09478673, 1.  ,       1.,
#   0. ,        0.03333333]]
# result = loaded_model.predict(X_train)
