

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

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