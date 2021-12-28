from typing import Sequence
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import matplotlib.pyplot as plt


# filename='d:\\py\\data\\hot-dog-places.csv'
filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed_transpose.csv"

# filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed.csv"
# "G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed.csv"
reader= pd.read_csv(filename)

# from_day=80
# to_day=120

# from_day=120
# to_day=180


# from_day=180
# to_day=200

from_day=300
to_day=350

# front_days=40
# x=reader.iloc[:,0]
# y1=reader.iloc[:,1]
# y2=reader.iloc[:,2]
# y3=reader.iloc[:,3]

# x=reader.iloc[:front_days,0]
# y1=reader.iloc[:front_days,1]
# y2=reader.iloc[:front_days,2]
# y3=reader.iloc[:front_days,3]


x=reader.iloc[from_day:to_day,0]
y1=reader.iloc[from_day:to_day,1]
y2=reader.iloc[from_day:to_day,2]
y3=reader.iloc[from_day:to_day,3]

# print("y1")
# print(y1)
# x=reader['Year']
# y1=reader['A']
# y2=reader['B']
# y3=reader['C']

# plt.bar(x,y1,align="center",color="b")
# plt.bar(x,y2,align="center",color="y")
# plt.bar(x,y3,align="center",color="m")
cyan="#CCFFFF"
yellow="#FFFF00"
purple="#CC99FF"
# plt.bar(x,y3,align="center",color=purple)
# plt.bar(x,y1,align="center",color=cyan)
# plt.bar(x,y2,align="center",color=yellow)
# 堆积柱状图 plt

# https://blog.csdn.net/sinat_38682860/article/details/91354235
# python get set 
class Bar:
    def __init__(self,x,y,color) -> None:
        self.x=x
        self.y=y
        self.color=color
    

# :list[Bar]
# :List[Bar]
# TypeError: 'type' object is not subscriptable
# https://zhuanlan.zhihu.com/p/150432111
def stacked_histogram(bars):

    # ys=[y1,y2,y3]
    last_y=0
    # for y in ys:
    #     plt.bar(x,y,bottom=last_y,align="center",color=cyan)
    #     last_y=y
    for bar in bars:
        plt.bar(bar.x,bar.y,bottom=last_y,align="center",color=bar.color)
        last_y=bar.y
    # plt.legend()

bars=[Bar(x,y1,cyan),Bar(x,y2,yellow),Bar(x,y3,purple)]
stacked_histogram(bars)
# plt.bar(x,y1,bottom=0,align="center",color=cyan)
# plt.bar(x,y2,bottom=y1,align="center",color=yellow)
# plt.bar(x,y3,bottom=y2,align="center",color=purple)


plt.legend()
name_base="美国疫情堆积柱状图"
name=f"{name_base}_从第{from_day}天到第{to_day}天"
# plt.title('2000-2010年热狗大胃王比赛成绩堆积柱状图',family ='kaiti',size=20, color ='g',loc='center')
plt.title(name,family ='kaiti',size=20, color ='#666699',loc='center')

# save_path=name+".jpg"
# save_path=f"{name}_从第{from_day}天到第{to_day}天.jpg"
save_path=f"{name}.jpg"
# save_path="d:\\py\\2000-2010年热狗大胃王比赛成绩堆积柱状"
plt.xticks(x, x, rotation=30)  # 这里是调节横坐标的倾斜度，rotation是度数
plt.savefig(save_path)
# plt 横坐标 斜着
# https://blog.csdn.net/zerow__/article/details/93635208

plt.show()

