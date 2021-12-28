

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from plotly.figure_factory._dendrogram import sch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif']=[u'SimHei']

# [1,1,1,1,1,22,2,2,,2,2]
# 这样的数据 把他 pie 
# lst_data 是一个 series吧
def pie_list(lst_data):
    labels=list(set(lst_data))
    fracs=[]
#     city_col=data["所在市"]
    for label in labels:
        fracs.append(lst_data[lst_data==label].shape[0])
    plt.pie(fracs,labels=labels)

def get_fracs_and_labels(lst_data):
    labels=list(set(lst_data))
    fracs=[]
    #     city_col=data["所在市"]
    for label in labels:
        fracs.append(lst_data[lst_data==label].shape[0])
    return fracs,labels

def bar_list(lst_data):
    fracs,labels=get_fracs_and_labels(lst_data)
    plt.bar(x=labels,height=fracs)

# 因为x轴如果数据太多的话 他们都会挤在一起,所以写了个函数,间隔几个再显示x轴上的标记
def show_lbls(ax,lbl_show_step):
    i=0
    for label in ax.get_xticklabels():
        # print("label")
        # print(label)
        if i==lbl_show_step:
            i=0
            continue
        label.set_visible(False)
        i+=1


def get_sum(type_name,state_filename_base):

    df = pd.read_excel(state_filename_base.format(type_name))
    # pd 每一列 求和
    df_sum = df.sum()
    # print("df_sum")
    # print(df_sum)
    return df_sum


def get_3type_df(state_filename_base):
    # df_confirmed = pd.read_excel(state_filename_base.format("confirmed"))
    # df_recovered = pd.read_excel(state_filename_base.format("recovered"))
    # df_death = pd.read_excel(state_filename_base.format("death"))
    # df=

    df=pd.DataFrame({})
    df["confirmed"]=get_sum("confirmed",state_filename_base)
    df["recovered"]=get_sum("recovered",state_filename_base)
    df["death"]=get_sum("death",state_filename_base)

    # df["confirmed"]=get_sum("confirmed")
    # df["recovered"]=get_sum("recovered")
    # df["death"]=get_sum("death")

    # print("df_death")
    # print(df_death)
    # print("df_death.shape")
    # print(df_death.shape)
    return df

# 同样的 折线图上面标记的数字 也是要隔几个显示一下 不然就叠在一起,不过具体要隔几个 还是要慢慢调出来
# 这里写了一个函数 方便调试和调用
def show_txt(x,y,show_txt_step,plt):
    i=0
    down=0.5
    # down=0.03
    for a,b in zip(x,y):
        if i==show_txt_step:
            # 可以考虑 在相近的 x y 的地方  不要再重复放置  text
            plt.text(a,b-down,b,ha='center',va='bottom',fontsize=11)  
            i=0
        i+=1


def hierarchicalClusteringTest(X):
    plt.figure(figsize=(20, 15))
    # dendrogram 系统树图（一种表示亲缘关系的树状图解）;
    # 连接; 联系; 链环; 连锁; 联动装置; linkage

    dendogram = sch.dendrogram(sch.linkage(X, method="ward"))
    # dendogram.

    plt.show()
#
# def  final_KMeans(X, n_clusters, df):
#
#     clf_final = KMeans(n_clusters=n_clusters, init='k-means++', random_state=6)
#     # clf_final = KMeans(n_clusters=3, init='k-means++', random_state=6)
#     clf_final.fit(X)
#
#     # 分类
#     df["Clusters"] = clf_final.predict(X)
#     df_lst=[]
#     for i in range(n_clusters):
#         df_lst.append(df[df["Clusters"] == i].head(15))
#     cluster_summary = pd.concat(df_lst)
#     # cluster_summary = pd.concat([df[df["Clusters"] == 1].head(15),
#     #                              df[df["Clusters"] == 2].head(15),
#     #                              df[df["Clusters"] == 0].head(15)])
#     # cluster_summary.style.background_gradient(cmap='Reds').format("{:.2f}")
#     return  cluster_summary



def  final_KMeans(X, n_clusters, df,head_num):

    clf_final = KMeans(n_clusters=n_clusters, init='k-means++', random_state=6)
    # clf_final = KMeans(n_clusters=3, init='k-means++', random_state=6)
    clf_final.fit(X)

    # 分类
    df["Clusters"] = clf_final.predict(X)
    df_lst=[]
    for i in range(n_clusters):
        if head_num is None:
            df_lst.append(df[df["Clusters"] == i].head(15))
        else:
            df_lst.append(df[df["Clusters"] == i].head(head_num))
    cluster_summary = pd.concat(df_lst)
    # cluster_summary = pd.concat([df[df["Clusters"] == 1].head(15),
    #                              df[df["Clusters"] == 2].head(15),
    #                              df[df["Clusters"] == 0].head(15)])
    # cluster_summary.style.background_gradient(cmap='Reds').format("{:.2f}")
    return  cluster_summary


def elbowMethod(X):
    wcss = []
    sil = []
    for i in range(2, 11):
        # 分类的个数 去尝试 每种尝试 发现 2 3 会好一些
        # 再根据层次聚类图 我们认为 选择分成3类比较好
        clf = KMeans(n_clusters=i, init='k-means++', random_state=42)
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


# 因为作图可能是做很多个折线图 就可能他们的x都是一样的,然后分别是好几个列来和这个x做图
# 于是写了一个函数 传入cols 是列名的列表,可以让x 和这些列来做折线图,同时可以在折线图上显示数字
def plot_lst(x,cols,df,show_txt_step,plt):
    for col in cols:
        # df.iloc
        plt.plot(x, df[col],label=col)
        # plt.plot(x, df[col],label=col)
        y=df[col]
        show_txt(x,y,show_txt_step,plt)

def plot_lst_by_idx_lst(x,idx_lst,df,show_txt_step,plt):
    col_lst=df.columns.tolist()
    for idx in idx_lst:
        # df.iloc
        y=df.iloc[:,idx]
        plt.plot(x,y,label=col_lst[idx])
        # plt.plot(x, df[col],label=col)
        # y=df[col]
        show_txt(x,y,show_txt_step,plt)

def bar_lst_plt(x,cols,df,show_txt_step,plt):
    for col in cols:
        plt.bar(x, df[col],label=col)
        y=df[col]
        show_txt(x,y,show_txt_step,plt)