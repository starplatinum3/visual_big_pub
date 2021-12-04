import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install plotly
try:
    # https://stackoverflow.com/questions/57105747/modulenotfounderror-no-module-named-plotly-graph-objects/57112843
    #     import plotly.graph_objects as go
    #     import plotly.express as px
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    from plotly import graph_objs as go
    from plotly import express as px
# import plotly.express as px
# import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime as dt
from datetime import timedelta
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
import statsmodels.api as sm
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
# from fbprophet import Prophet
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.stattools import adfuller
# !pip install pyramid-arima
# from pyramid.arima import auto_arima
std=StandardScaler()
#pd.set_option('display.float_format', lambda x: '%.6f' % x)
# out
# filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed_transpose.csv"

state_filename_base=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\State_level_summary\US_State_summary_covid19_{}_trpo.xlsx"
# state_filename_base=r"COVID-19-Data-master\US\State_level_summary\US_State_summary_covid19_{}_trpo.xlsx"
# state_filename_base=r"COVID-19-Data-master/US/State_level_summary/US_State_summary_covid19_{}_trpo.xlsx"

def get_sum(type_name):

    df = pd.read_excel(state_filename_base.format(type_name))
    # pd 每一列 求和
    df_sum = df.sum()
    # print("df_sum")
    # print(df_sum)
    return df_sum


def get_3type_df():
    # df_confirmed = pd.read_excel(state_filename_base.format("confirmed"))
    # df_recovered = pd.read_excel(state_filename_base.format("recovered"))
    # df_death = pd.read_excel(state_filename_base.format("death"))
    # df=

    df=pd.DataFrame({})
    df["confirmed"]=get_sum("confirmed")
    df["recovered"]=get_sum("recovered")
    df["death"]=get_sum("death")
    # print("df_death")
    # print(df_death)
    # print("df_death.shape")
    # print(df_death.shape)
    return df
covid=get_3type_df()

confirmed_col="confirmed"
recovered_col="recovered"
death_col="death"

datewise=covid
yy=datewise[confirmed_col]-datewise[recovered_col]-datewise[death_col]
print("yy")
print(yy)
print("datewise.index")
print(datewise.index)
# 总的 和
# 根据不同的 couty
# 确诊的 - 治愈的 - 死亡的
# 就是现在还在患病的
# 分配  Distribution 分布
fig=px.bar(x=datewise.index,y=datewise[confirmed_col]-datewise[recovered_col]-datewise[death_col])
fig.update_layout(title="Distribution of Number of Active Cases 累计患病的分布(各个县)",
                  xaxis_title="县",yaxis_title="Number of Cases 患病的个数",)
# xaxis_title="Date",yaxis_title="Number of Cases",
fig.show()
# 正在患病的分布(各个县)"
# 为什么没有显示呢

# countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100
# countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100