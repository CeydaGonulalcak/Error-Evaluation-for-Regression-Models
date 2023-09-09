import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_excel("ML1/data.xlsx", sheet_name="reg")
df.head()


### Maaş Tahmin ###
# y' = b + wx
# b = 275
# w = 90
def my_linear_regression(x, bias=275, weight=90):
    return bias + (weight * x)

df["Maaş Tahmin"] = df["Deneyim Yılı"].apply(my_linear_regression)

df["Maaş Tahmin"]


### Hata ###
def my_error(y, yi):
    return y - yi

df["Hata"] = df[["Maaş", "Maaş Tahmin"]].apply(lambda x: my_error(x[0], x[1]), axis=1)

df["Hata"]


### Hata Kareler ###
def my_root_error(y, yi):
    return (y - yi) ** 2

df["Hata Kareler"] = [my_root_error(row["Maaş"], row["Maaş Tahmin"]) for idx,row in df.iterrows()]

df["Hata Kareler"]


### Mutlak Hata ###

df["Mutlak Hata"] = abs(df["Maaş Tahmin"] - df["Maaş"])

df["Mutlak Hata"]



#mse
mse = sum(df["Hata"] ** 2) / len(df)
mse

#rmse
rmse = math.sqrt(mse)
rmse

#mae
mae = sum(abs(df["Hata"])) / len(df)
mae




mean_squared_error(df["Maaş"], df["Maaş Tahmin"])
math.sqrt(mean_squared_error(df["Maaş"], df["Maaş Tahmin"]))
mean_absolute_error(df["Maaş"], df["Maaş Tahmin"])



################################################

df.head()
lin = LinearRegression().fit(df[["Deneyim Yılı"]], df[["Maaş"]])
lin.coef_
lin.intercept_


np.inner(lin.coef_, df[["Deneyim Yılı"]].iloc[0,:]) + lin.intercept_
df.iloc[0,:]

