import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_excel("CaseStudy1/data.xlsx", sheet_name="reg")
df.head()


### Salary Prediction ###
# y' = b + wx
# b = 275
# w = 90

def my_linear_regression(x, bias=275, weight=90):
    return bias + (weight * x)

df["Salary Prediction"] = df["Years of Experience"].apply(my_linear_regression)

df["Salary Prediction"]


### Error ###

def my_error(y, yi):
    return y - yi

df["Error"] = df[["Salary", "Salary Prediction"]].apply(lambda x: my_error(x[0], x[1]), axis=1)

df["Error"]


### Error Squares ###

def my_root_error(y, yi):
    return (y - yi) ** 2

df["Error Squares"] = [my_root_error(row["Salary"], row["Salary Prediction"]) for idx,row in df.iterrows()]

df["Error Squares"]


### Absolute Error ###

df["Absolute Error"] = abs(df["Salary Prediction"] - df["Salary"])

df["Absolute Error"]


#mse
mse = sum(df["Error"] ** 2) / len(df)

#rmse
rmse = math.sqrt(mse)

#mae
mae = sum(abs(df["Error"])) / len(df)


mean_squared_error(df["Salary"], df["Salary Prediction"])
math.sqrt(mean_squared_error(df["Salary"], df["Salary Prediction"]))
mean_absolute_error(df["Salary"], df["Salary Prediction"])


df.head()
lin = LinearRegression().fit(df[["Years of Experience"]], df[["Salary"]])
lin.coef_
lin.intercept_


np.inner(lin.coef_, df[["Salary Prediction"]].iloc[0,:]) + lin.intercept_
df.iloc[0,:]

