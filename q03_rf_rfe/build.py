# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    features=X.columns
    rfe=RFE(RandomForestClassifier(),len(features)/2)
    rfe=rfe.fit(X,y)
#     print(rfe.n_features_)
#     print(rfe.support_)
#     print(rfe.ranking_)
    return features[rfe.get_support()].tolist()
print(rf_rfe(data))



