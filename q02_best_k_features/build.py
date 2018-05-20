# %load q02_best_k_features/build.py
# Default imports
import numpy as np
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(data,k=20):
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    f_test, _ = f_regression(X, y)
    fs=SelectPercentile(percentile =k)
    X_fs=fs.fit_transform(X,y)
    col=np.asarray(X.columns.values)
    support=np.asarray(fs.get_support())
    col_with_support=col[support]
    fs_score = list(zip(features,scores))
    df = pd.DataFrame(fs_score,columns=['Name','Score'])
    return df.sort_values(['Score','Name'],ascending = [False,True])['Name'].tolist() 

    return col_with_support

    sp = SelectPercentile(f_regression,percentile=k)
    sp.fit_transform(X,y)
    features = X.columns.values[sp.get_support()]
    scores = sp.scores_[sp.get_support()]
    fs_score = list(zip(features,scores))
    df = pd.DataFrame(fs_score,columns=['Name','Score'])
    return df.sort_values(['Score','Name'],ascending = [False,True])['Name'].tolist() 

print(percentile_k_features(data))



