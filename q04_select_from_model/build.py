# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    features=X.columns
    clf=RandomForestClassifier(random_state=9)
    model = SelectFromModel(clf)
    model.fit_transform(X,y)
    return features[model.get_support()].tolist()
print(select_from_model(data))


