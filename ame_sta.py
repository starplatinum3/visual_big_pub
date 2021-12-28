import time
start=time.time()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime

end=time.time()
print("load moudle cost {:.3f} s".format(end-start))

#-----------------------------------(a) 堆积面积图--------------------------------------------
# filename='d:\\py\\data\\SaleStackedArea_Data.csv'
# filename='SaleStackedArea_Data.csv'
filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed_transpose.csv"
df=pd.read_csv(filename,index_col =0)
df.index=[datetime.strptime(d, '%Y/%m/%d').date() for d in df.index]

# load moudle cost 18.229 s
# Sum_df
# North_America    2747.5
# Europe           2521.2
# Asia             1633.5
# South_America    1119.7
# Africa            582.7
# dtype: float64

Sum_df=df.apply(lambda x: x.sum(), axis=0).sort_values(ascending=False)
# 求和是 一列的和吗
# print("Sum_df")
# print(Sum_df)

df=df[Sum_df.index]
# df=df.loc[0]
# df=df.loc[[0,1,2,3,4]]
# df=df[[0,1,2,3,4]]
# https://www.jianshu.com/p/8024ceef4fe2
# df= df.loc[:,[0,1,2,3,4] ]
front_cnt=4
# 具体的碰到的问题,其实是一个小细节,就是怎么取得df的前面几列
# 虽然很简单的问题 但是网上查不到. 我这方面也没什么基础,所以其实是查了大量资料,到处找才找到的
# 网上虽然 用列索引取出列 这种操作是挺多的 但是用数字取出 却比较少见
# 但是最终还是找到了
# https://blog.csdn.net/houyanhua1/article/details/87809185
# 应该类似这样找
# print(df.iloc[1:3,1:3])

# 这个东西挺花费时间的
df= df.iloc[:,:front_cnt]
# df 取出前面4列
# df 取出第一列
# 至于为什么要取出前面几列 是因为 数据太多了 要是所有都拿来作图的话  图会有很多 这样大家之间都不好分辨
# 因此是分开几个作图


# df=df.loc[0,1,2,3,4]
# df 取出 前面 10列
# https://zhidao.baidu.com/question/438977714866059404.html

columns=df.columns

# df.index.values
# etime.date(2011, 5, 30)
#  datetime.date(2011, 6, 30) datetime.date(2011, 7, 30)
#  datetime.date(2011, 8, 30) datetime.date(2011, 9, 30)
#  datetime.date(2011, 10, 30) datetime.date(2011, 11, 30)
#  datetime.date(2011, 12, 30)]
#  df.values.T
# [[23.  19.9 21.  22.1 25.1 27.  19.8 19.9 19.4 21.6 24.4 24.7 23.5 23.4
#   24.6 24.9 27.4 26.5 26.7 27.1 28.7 31.6 30.9 31.6 28.7 26.9 27.3 34.4
#   34.8 35.6 33.6 32.  31.2 31.1 29.7 31.5 31.8 33.  31.3 33.1 32.8 33.5
#   27.  27.4 27.4 26.3 26.4 26.8 23.9 23.2 23.  28.  29.1 28.6 22.8 22.4
#   22.3 17.6 16.8 16.2 18.7 18.8 19.

colors= sns.husl_palette(len(columns),h=15/360, l=.65, s=1).as_hex() 

fig =plt.figure(figsize=(5,4), dpi=100)

# print("df.index.values")
# print( df.index.values)
# print(" df.values.T")
# print( df.values.T)
plt.stackplot(df.index.values, 
              df.values.T,alpha=1, labels=columns, linewidth=1,edgecolor ='k',colors=colors)
# confirmed persons
xlabel="Year"
ylabel="confirmed persons"
plt.xlabel(xlabel)
plt.ylabel(ylabel)

# plt.xlabel("Year")
# plt.ylabel("Value")

plt.legend(title="group",loc="center right",bbox_to_anchor=(1.5, 0, 0, 1),edgecolor='none',facecolor='none')
theme="US_County_summary_covid19_confirmed_transpose"
# title="{}_front{}places".format(theme,front_cnt)
title=f"{theme}_front{front_cnt}places"
plt.title(title)
plt.legend()
plt.show()

#---------------------------------(b)百分比堆积面积图---------------------------------------------
# df=pd.read_csv('d:\\py\\data\\SaleStackedArea_Data.csv',index_col =0)
df=pd.read_csv(filename,index_col =0)
df= df.iloc[:,:front_cnt]
df.index=[datetime.strptime(d, '%Y/%m/%d').date() for d in df.index]
SumRow_df=df.apply(lambda x: x.sum(), axis=1)
df=df.apply(lambda x: x/SumRow_df, axis=0)
meanCol_df=df.apply(lambda x: x.mean(), axis=0).sort_values(ascending=False)
df=df[meanCol_df.index]
columns=df.columns

colors= sns.husl_palette(len(columns),h=15/360, l=.65, s=1).as_hex() 

fig =plt.figure(figsize=(5,4), dpi=100)
plt.stackplot(df.index.values, df.values.T,labels=columns,colors=colors,
              linewidth=1,edgecolor ='k')

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.legend(title="group",loc="center right",bbox_to_anchor=(1.5, 0, 0, 1),edgecolor='none',facecolor='none')
title=f"{theme}_front{front_cnt}places_percentage"
plt.title(title)
plt.legend()
plt.show()