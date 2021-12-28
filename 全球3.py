import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# !pip install plotly
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
# import datetime as dt
# from datetime import timedelta
# from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error, r2_score
# import statsmodels.api as sm
# from statsmodels.tsa.api import Holt, SimpleExpSmoothing, ExponentialSmoothing
# # from fbprophet import Prophet
# from sklearn.preprocessing import PolynomialFeatures
# from statsmodels.tsa.stattools import adfuller

# !pip install pyramid-arima
# from pyramid.arima import auto_arima
std = StandardScaler()
# pd.set_option('display.float_format', lambda x: '%.6f' % x)
# out
# filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed_transpose.csv"

# state_filename_base = r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\State_level_summary\US_State_summary_covid19_{}_trpo.xlsx"
# state_filename_base=r"COVID-19-Data-master\US\State_level_summary\US_State_summary_covid19_{}_trpo.xlsx"
# state_filename_base=r"COVID-19-Data-master/US/State_level_summary/US_State_summary_covid19_{}_trpo.xlsx"

# state_filename_base =r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\China\Province_level_summary\China_Province_summary_covid19_{}_trpo.xlsx"
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
state_filename_base =r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\Global\country_level_summary\Global_summary_covid19_confirmed.csv"
# class  ClustersAnalysis:
#     def __init__(self):

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

    df = pd.DataFrame({})
    df["confirmed"] = get_sum("confirmed")
    df["recovered"] = get_sum("recovered")
    df["death"] = get_sum("death")
    # print("df_death")
    # print(df_death)
    # print("df_death.shape")
    # print(df_death.shape)
    return df


covid = get_3type_df()

confirmed_col = "confirmed"
recovered_col = "recovered"
death_col = "death"

datewise = covid
# yy=datewise[confirmed_col]-datewise[recovered_col]-datewise[death_col]
# print("yy")
# print(yy)
# print("datewise.index")
# print(datewise.index)
# 总的 和
# 根据不同的 couty
# 确诊的 - 治愈的 - 死亡的
# 就是现在还在患病的
# 分配  Distribution 分布
countrywise = datewise
# countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100
# countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100

countrywise["Mortality"] = (countrywise[death_col] / countrywise[confirmed_col]) * 100
countrywise["Recovery"] = (countrywise[recovered_col] / countrywise[confirmed_col]) * 100

# fig=px.bar(x=datewise.index,y=datewise[confirmed_col]-datewise[recovered_col]-datewise[death_col])
# fig.update_layout(title="Distribution of Number of Active Cases 累计患病的分布(各个县)",
#                   xaxis_title="县",yaxis_title="Number of Cases 患病的个数",)
# # xaxis_title="Date",yaxis_title="Number of Cases",
# fig.show()

# 正在患病的分布(各个县)"
# 为什么没有显示呢


# countrywise.drop(index=(countrywise.loc[(countrywise['table']=='sc')].index))
# countrywise.drop(index="Unknown",replace=True)
countrywise=countrywise.drop(index="Unknown")
# print("countrywise.index")
# print(countrywise.index)
# print("countrywise")
# print(countrywise)
# countrywise[]
# wcss = []
# sil = []
# for i in range(2, 11):
#     # 分类的个数 去尝试 每种尝试 发现 2 3 会好一些
#     # 再根据层次聚类图 我们认为 选择分成3类比较好
#     clf = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     clf.fit(X)
#     labels = clf.labels_
#     centroids = clf.cluster_centers_
#     sil.append(silhouette_score(X, labels, metric='euclidean'))
#     wcss.append(clf.inertia_)
#

X = countrywise[["Mortality", "Recovery"]]
# 死亡率 Mortality
# Standard Scaling since K-Means Clustering is a distance based alogrithm
# 标准缩放，因为K-均值聚类是一种基于距离的算法
X = std.fit_transform(X)
# pd 删掉某一行数据

# 关于肘部法则
def ElbowMethod(X):
    wcss = []
    sil = []
    for i in range(2, 11):
        # 分类的个数 去尝试 每种尝试 发现 2 3 会好一些
        # 再根据层次聚类图 我们认为 选择分成3类比较好
        clf = KMeans(n_clusters=i, init='k-means++', random_state=42)
        # 分类的个数是2..11  我们都去尝试一下,fit 一下,看一下的出来的结果,画个图,直观一点
        # 看看那种方案好点
        clf.fit(X)
        labels = clf.labels_
        centroids = clf.cluster_centers_
        sil.append(silhouette_score(X, labels, metric='euclidean'))
        wcss.append(clf.inertia_)
    x = np.arange(2, 11)
    plt.figure(figsize=(10, 5))
    plt.plot(x, wcss, marker='o')
    plt.xlabel("Number of Clusters 集群的个数 ")
    # 集群;群集;
    plt.ylabel("Within Cluster Sum of Squares (WCSS) 簇内平方和")
    # 簇内平方和（WCSS）
    plt.title("Elbow Method 肘部法则")
    # –Elbow Method和轮廓...
    # 肘部法则
    plt.show()


# countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100
# countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100

import scipy.cluster.hierarchy as sch


# 等级制度(尤指社会或组织); 统治集团; 层次体系; hierarchy
#

def HierarchicalClusteringTest(X):
    plt.figure(figsize=(20, 15))
    # dendrogram 系统树图（一种表示亲缘关系的树状图解）;
    # 连接; 联系; 链环; 连锁; 联动装置; linkage

    dendogram = sch.dendrogram(sch.linkage(X, method="ward"))
    # dendogram.

    plt.show()


def  final_KMeans(X,n_clusters):

    clf_final = KMeans(n_clusters=n_clusters, init='k-means++', random_state=6)
    # clf_final = KMeans(n_clusters=3, init='k-means++', random_state=6)
    clf_final.fit(X)

    # 分类
    countrywise["Clusters"] = clf_final.predict(X)

    cluster_summary = pd.concat([countrywise[countrywise["Clusters"] == 1].head(15),
                                 countrywise[countrywise["Clusters"] == 2].head(15),
                                 countrywise[countrywise["Clusters"] == 0].head(15)])
    cluster_summary.style.background_gradient(cmap='Reds').format("{:.2f}")
    return  cluster_summary

    # 背景梯度
    # plt.show()
    # cluster_summary
    # print("cluster_summary")
    # print(cluster_summary)
    # 数据显示  治愈率是0 这是
    # 我们把这些州 按照治愈率和 死亡率 分成了三类
    # 根据背景梯度图 显示 一类是死亡率较高 0-死亡率其次 , 2-死亡率最低


# clf_final = KMeans(n_clusters=3, init='k-means++', random_state=6)
# clf_final.fit(X)
#
# # 分类
# countrywise["Clusters"] = clf_final.predict(X)
#
# cluster_summary = pd.concat([countrywise[countrywise["Clusters"] == 1].head(15),
#                              countrywise[countrywise["Clusters"] == 2].head(15),
#                              countrywise[countrywise["Clusters"] == 0].head(15)])
# cluster_summary.style.background_gradient(cmap='Reds').format("{:.2f}")
# # 背景梯度
# # plt.show()
# print("cluster_summary")
# print(cluster_summary)
# 数据显示  治愈率是0 这是
# 我们把这些州 按照治愈率和 死亡率 分成了三类
# 根据背景梯度图 显示 一类是死亡率较高 0-死亡率其次 , 2-死亡率最低

# 显示分成三类 好一点
# ElbowMethod(X)
# 显示分成 3类
# HierarchicalClusteringTest(X)

# final_KMeans(X,3)
# unkown 的城市其实 属于是 没有用处的数据,那么 把他删掉吧

# ElbowMethod(X)
HierarchicalClusteringTest(X)

# def separate(countrywise):
#     plt.figure(figsize=(10,5))
#     sns.scatterplot(x=countrywise["Recovery"],y=countrywise["Mortality"],hue=countrywise["Clusters"],s=100)
#     plt.axvline(((datewise["Recovered"]/datewise["Confirmed"])*100).mean(),
#                 color='red',linestyle="--",label="Mean Recovery Rate around the World")
#     plt.axhline(((datewise["Deaths"]/datewise["Confirmed"])*100).mean(),
#                 color='black',linestyle="--",label="Mean Mortality Rate around the World")
#     plt.legend()

def separate(countrywise):
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=countrywise["Recovery"],y=countrywise["Mortality"],hue=countrywise["Clusters"],s=100)
    # 这里写的 是这个国家的 数据 所以都在线上 所以有问题
    plt.axvline(countrywise["Recovery"].mean(),color='red',linestyle="--",label="Mean Recovery Rate around the World")
    plt.axhline(countrywise["Mortality"].mean(),
                color='black',linestyle="--",label="Mean Mortality Rate around the World")
    plt.legend()

separate(countrywise)