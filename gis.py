
import matplotlib.pyplot as plt
import pandas as pd
import plotly

try: 
    # https://stackoverflow.com/questions/57105747/modulenotfounderror-no-module-named-plotly-graph-objects/57112843
    import plotly.graph_objects as go
except ImportError as e:
    from plotly import graph_objs as go

# import plotly.graph_objects as go
import seaborn as sns

# sns.set_style("darkgrid",{'axes.facecolor': '.95'})
# sns.set_context("notebook", font_scale=1.5,
#                 rc={'axes.labelsize': 13, 'legend.fontsize':13, 
#                     'xtick.labelsize': 12,'ytick.labelsize': 12})

# https://blog.csdn.net/u010472607/article/details/82789887
# import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 修改读入的文件名
# airports_filename="airports.csv"
# filename=r"D:\download\GIS疫情地图2020全年-至今数据\【GIS点滴疫情地图·2020年01月02日-2021年01月25日】国内每天疫情统计.xlsx"
filename=r"D:\download\GIS疫情地图2020全年-至今数据\【GIS点滴疫情地图·2020年01月21日-2021年01月25日】国内每日新增疫情数据.xlsx"
airports_filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed_transpose.csv"
# us_airports_df = pd.read_csv('D:\\py\\data\\airports.csv')
state_filename_base =r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\Global\country_level_summary\Global_summary_covid19_confirmed.csv"

# state_filename_base=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\State_level_summary\US_State_summary_covid19_{}_trpo.xlsx"
# state_filename_base="G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\State_level_summary\US_State_summary_covid19_confirmed_trpo.xlsx"

# df = pd.read_csv(airports_filename)
# df = pd.read_csv(state_filename_base)
df = pd.read_excel(filename)
# pd 每一列 求和
# df_sum = df.sum()
# print("df_sum")
# print(df_sum)
# df.plot()
# plt.show()

def read_file(type_name):

    df = pd.read_excel(state_filename_base.format(type_name))
    # pd 每一列 求和 
    df_sum = df.sum()
    print("df_sum")
    print(df_sum)
    return df

def get_sum(type_name):

    df = pd.read_excel(state_filename_base.format(type_name))
    # pd 每一列 求和 
    df_sum = df.sum()
    # print("df_sum")
    # print(df_sum)
    return df_sum

# df=read_file("confirmed")
# print("df")
# print(df)

# Pairs Plots

# df_confirmed = pd.read_excel(state_filename_base.format("confirmed"))
# df_recovered = pd.read_excel(state_filename_base.format("recovered"))
# df_death = pd.read_excel(state_filename_base.format("death"))
# # df=

# df=pd.DataFrame({})
# df["confirmed"]=get_sum("confirmed")
# df["recovered"]=get_sum("recovered")
# df["death"]=get_sum("death")
# print("df_death")
# print(df_death)
# print("df_death.shape")
# print(df_death.shape)
# # df["death"]=df_death
# # df["recovered"]=df_recovered
# # df["confirmed"]=df_confirmed
# print("df")
# print(df)

# # df 前面 几行
# # print("df[:5,:]")
# # print(df[:5,:])
# df_first5_lines=df[:5]
# # df= df.iloc[:,:4 ]
# # g=sns.pairplot(df[:5,:], height =2)
# g=sns.pairplot(df_first5_lines, height =2)
# # import color
# from color import Color
# # color.Color.black
# edgecolor=Color.BlueViolet

# # g = g.map_diag(plt.hist,color='#00C07C',density=False,edgecolor="k",bins=10,alpha=0.8,linewidth=0.5)
# # g = g.map_offdiag(plt.scatter, color='#00C2C2',edgecolor="k", s=30, linewidth=0.25)
# # 是为了看看每个参数的相关性吗
# # 四个维度的数据 全都画画
# g = g.map_diag(plt.hist,color=Color.SkyBlue,density=False,edgecolor=edgecolor,bins=10,alpha=0.8,linewidth=0.5)
# g = g.map_offdiag(plt.scatter, color=Color.DeepPink,edgecolor=edgecolor, s=30, linewidth=0.25)

# plt.subplots_adjust(hspace=0.05, wspace=0.05)
# # pdf_name='Matrix_Scatter2.pdf'
# pdf_name='Matrix_Scatter2_modi.pdf'
# g.savefig(pdf_name)
# # OSError: [Errno 22] Invalid argument: 'Matrix_Scatter_Singledata.png'
# png_name="Matrix_Scatter_Singledata_modi.png"
# # png_name="Matrix_Scatter_Singledata.png"
# plt.savefig(png_name)

#--------------------------------(b) 多数据系列------------------------------------------------------
# g= sns.pairplot(df, kind = 'scatter', hue="variety", palette="Set1", height=2.5, plot_kws=dict(s=50, alpha=0.4))
# #hue: name of variable in data

# g = g.map_diag(sns.kdeplot, lw=1, legend=False)

# g = g.map_offdiag(plt.scatter, edgecolor="k", s=30,linewidth=0.2)


# plt.subplots_adjust(hspace=0.05, wspace=0.05)

#plt.savefig("Matrix_Scatter_Multidata.png")



# 文档：df_sum = df.sum().note
# 链接：http://note.youdao.com/noteshare?id=fb819d91c744852d75d31fbc82c692bf&sub=D9C9E22A37584F8997A88415A8C8ABCD


# 每一列的数据都给他求和了 ，也就是这些天的数据知道了，关于每个县
# us_airports_df.head()
# 然后每个 种类的 都给他求和
# 行数 太多了 但是可以只画图 四行
# 虽然其实感觉没有用 但是为了凑出 图片。。
# https://www.cnblogs.com/hello-/articles/9614765.html
county='Osage County'
# x=df['date']
# y= df[county]
# df=us_airports_df
# 根据state 来一个和
# df[]

fig, ax = plt.subplots(1, 1)
countys=['Osage County',"Oswego County","Park County","Pettis County","Platte County"]


# show_txt_step=10
show_txt_step=20
def show_txt(x,y,show_txt_step):
    i=0
    down=0.5
    # down=0.03
    for a,b in zip(x,y):
        if i==show_txt_step:
            # 可以考虑 在相近的 x y 的地方  不要再重复放置  text
            plt.text(a,b-down,b,ha='center',va='bottom',fontsize=11)  
            i=0
        i+=1


# for ct in countys:
#     plt.plot(x, df[ct],label=ct)
#     y=df[ct]
#     show_txt(x,y,show_txt_step)

import util
# https://blog.csdn.net/AI_future/article/details/103301152
# ax.plot(x, y)
# plt.plot(x, y)
# plt.plot(x, df["Oswego County"])
# plt.plot(us_airports_df['date'], us_airports_df[county])

# https://blog.csdn.net/qq_41185868/article/details/89226991
# No handles with labels found to put in legend.
# 不是legend
# 不要全部显示  plt.legend()
# 太多 plt.legend()
# x轴 太多 plt
# title_base="US_County_summary_covid19_confirmed_transpose"
# title_base="中国随着时间确诊图"
title_base="中国关于时间疫情状况折线图"
# type="折线图"
type="柱状图"
# title=f"{title_base}_{county}_{type}"
# title=f"{title_base}_{type}_some_countys"
title=f"{title_base}"
# plt 标记数字
plt.title(title)

# show_txt(x,y,show_txt_step)
# plt.show()
# https://blog.csdn.net/qq_29721419/article/details/71638912
# https://blog.csdn.net/qq_29721419/article/details/71638912
# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
x=df["day"]
y=df["diagnose"]

# plt.bar(x, y,label="确诊")
# 柱状图

# plt.plot(x, y,label="确诊")
# plt.plot(x, df["suspect"],label="疑似")
# plt.plot(x, df["cure"],label="治愈")
# plt.plot(x, df["death"],label="死亡")
# plt.plot(x, df["seriousCount"],label="严重")
# plt.plot(x, df["currentdiagnose"],label="目前确诊")
# 折线图
# plt.bar(x, df["diagnose"],label="确诊")

#
plt.scatter(x, y,label="确诊")
plt.scatter(x, df["suspect"],label="疑似")
plt.scatter(x, df["cure"],label="治愈")
plt.scatter(x, df["death"],label="死亡")
plt.scatter(x, df["seriousCount"],label="严重")
plt.scatter(x, df["currentdiagnose"],label="目前确诊")


# 这个花不了
# ValueError: `bins` must increase monotonically, when an array
# plt.hist(x, y,label="确诊")
# plt.hist(x, df["suspect"],label="疑似")
# plt.hist(x, df["cure"],label="治愈")
# plt.hist(x, df["death"],label="死亡")
# plt.hist(x, df["seriousCount"],label="严重")
# plt.hist(x, df["currentdiagnose"],label="目前确诊")

# fig, ax = plt.bar(x, y)
# util.show_lbls(ax,5)
# util.show_lbls(ax,40)
util.show_txt(x,y,50,plt)
util.show_lbls(ax,80)
# num_list = [1.5,0.6,7.8,6]
# plt.bar(range(len(num_list)), num_list)
# util.bar_lst_plt(x,countys,df,50,plt)

# util.bar_lst_plt(x,y,df,50,plt)

# plt.bar(x, y,label=county)
# util.show_lbls(ax,50)

plt.legend()
# 放在后面 才有 labels
# 不然 No handles with labels found to put in legend.
plt.show()


# import util
# # AttributeError: 'int' object has no attribute 'shape'
# util.bar_list([1,1,21,3,1,3,1,312,3,21])