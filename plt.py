
import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.graph_objects as go

# https://blog.csdn.net/u010472607/article/details/82789887
# import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 修改读入的文件名
# airports_filename="airports.csv"
airports_filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed_transpose.csv"
# us_airports_df = pd.read_csv('D:\\py\\data\\airports.csv')
us_airports_df = pd.read_csv(airports_filename)
filename_china_xlsx=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\China\City_level_summary\China_City_summary_covid19_confirmed_trpo.xlsx"
df_china = pd.read_excel(filename_china_xlsx)

# us_airports_df.head()
# https://www.cnblogs.com/hello-/articles/9614765.html
county='Osage County'
x=us_airports_df['date']
y= us_airports_df[county]
df=us_airports_df

fig, ax = plt.subplots(1, 1)
countys=['Osage County',"Oswego County","Park County","Pettis County","Platte County"]
citys=["Anshan","Xi'an","Langfang","Yuxi"]

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

def plot_lst(x,cols,df,show_txt_step):
    for col in cols:
        plt.plot(x, df[col],label=col)
        y=df[col]
        show_txt(x,y,show_txt_step)

plot_lst(x,cols=countys,df=df,show_txt_step=show_txt_step)
# for ct in countys:
#     plt.plot(x, df[ct],label=ct)
#     y=df[ct]
#     show_txt(x,y,show_txt_step)


# https://blog.csdn.net/AI_future/article/details/103301152
# ax.plot(x, y)
# plt.plot(x, y)
# plt.plot(x, df["Oswego County"])
# plt.plot(us_airports_df['date'], us_airports_df[county])
plt.legend()
# https://blog.csdn.net/qq_41185868/article/details/89226991
# No handles with labels found to put in legend.
# 不是legend
# 不要全部显示  plt.legend()
# 太多 plt.legend()
# x轴 太多 plt
title_base="US_County_summary_covid19_confirmed_transpose"
type="折线图"
# title=f"{title_base}_{county}_{type}"
title=f"{title_base}_{type}_some_countys"
# plt 标记数字
plt.title(title)

# 不用draw
# plt.draw()
# http://www.cocoachina.com/articles/98823
lbl_show_step=50
# lbl_show_step=10

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

# 没有隐藏啊
# lbl_show_step=20
# lbls=ax.get_xticklabels()
# ax.get_xticklabels() 都是空的
# print("lbls")
# print(lbls)
# ), Text(0, 0, ''), Text(0, 0, '')]

# for label in ax.get_xticklabels():
#     # print("label")
#     # print(label)
#     label.set_visible(False)
# 这样之后 貌似ax.get_xticklabels(): 就拿不到东西了

show_lbls(ax,lbl_show_step)
# show_lbls=ax.get_xticklabels()[::lbl_show_step]
# show_lbls=ax.get_xticklabels()
# # 无了
# print("show_lbls")
# print(show_lbls)

# for label in ax.get_xticklabels()[::lbl_show_step]:
#     # 没有东西吗
#     print("label")
#     print(label)
#     label.set_visible(True)

# plt.title(date)
# plt.savefig("{}.jpg".format(date), dpi=500)



# show_txt(x,y,show_txt_step)
plt.show()
# https://blog.csdn.net/qq_29721419/article/details/71638912

# plt.
# import util
# # AttributeError: 'int' object has no attribute 'shape'
# util.bar_list([1,1,21,3,1,3,1,312,3,21])