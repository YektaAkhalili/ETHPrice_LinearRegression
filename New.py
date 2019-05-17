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
New_Add = data[["ETH","Market Cap"]]

#should check if the new data is highly collinear with others...
New_Add = sm.add_constant(X, prepend=False)

print(New_Add.columns.get_loc("ETH"))
print(New_Add.columns.get_loc("Volume"))
print(New_Add.columns.get_loc("Market Cap"))

#to know if this "feature" should be added or not
vif_ETH = variance_inflation_factor(np.array(New_Add),0)
vif_Vol = variance_inflation_factor(np.array(New_Add),1)
vif_MC = variance_inflation_factor(np.array(New_Add),1)
print(vif_ETH, vif_MC)

#Model "New"
modelNew = sm.OLS(np.log(Y), np.log(New_Add))
resultNew = modelNew.fit()

#errors for this model
errors = resultNew.resid
print(resultNew.summary())

#predicting the prices using this model
predictions = resultNew.predict(Y)
print(predictions)

#visualize errors - looks normal! 
plt.hist(errors)
plt.title("Errors of Model ")
plt.show()

# save results in a png file ...
from PIL import Image, ImageDraw, ImageFont
image = Image.new('RGB', (800, 400))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 16)
draw.text((0, 0), str(resultNew.summary()), font=font)
image = image.convert('1') # bw
image = image.resize((600, 300), Image.ANTIALIAS)
image.save('output_New.png')
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(resultNew.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('output_New.png')
