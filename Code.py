#Code written by YektaAkhalili
#This code is a simple Linear Regression fit to ETH data

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_excel("New.xlsx")
X = data["Volume"]
Y = data["Price"]
X = sm.add_constant(X, prepend=False)
model = sm.OLS(Y,X)
results = model.fit()
#predicting the prices using this model
predictions = results.predict(X)
print(predictions.head())

plt.plot(X,Y, color='r')
plt.plot(predictions, Y, label='Predicted')
plt.xlabel("Volume")
plt.ylabel("Price")
plt.legend()
plt.show()
# plt.legend(loc="best")
