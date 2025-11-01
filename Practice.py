import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mlt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

df = pd.read_csv("C:/Users/prati/OneDrive/Desktop/python libraries/WineQT.csv")
# print(df.duplicated())
# print(df.info())
# print(df.describe())
# print(df.head())
X = df.drop("quality", axis=1)
y = df["quality"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_split=5,min_samples_leaf=2,max_features='sqrt',n_jobs=-1, random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
ACC = accuracy_score(y_test,y_pred)
print(y_pred)
print(ACC)