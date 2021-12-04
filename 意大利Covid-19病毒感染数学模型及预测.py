import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
# %matplotlib inline


# url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
# df =pd.read_csv(url)
filename="dpc-covid19-ita-andamento-nazionale.csv"
df =pd.read_csv(filename)


df =df.loc[:,['data','totale_casi']]
FMT ='%Y-%m-%d %H:%M:%S'
date =df['data']
df['data']= date.map(lambda x : (datetime.strptime(x, FMT) -datetime.strptime("2020-01-01 00:00:00", FMT)).days  )


# https://zhuanlan.zhihu.com/p/144353126
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

x =list(df.iloc[:,0])
y =list(df.iloc[:,1])
# 但是他的a 是怎么来的 速度
fit = curve_fit(logistic_model,x,y,p0=[2,100,20000])