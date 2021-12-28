# -*- coding: utf-8 -*-
import time
start=time.time()
import pandas as pd
import numpy as np
from plotnine import *

import matplotlib.pyplot as plt 

import calmap
end=time.time()
# load moudle cost 0.007094840288162232 s
# read file cost 0.000589686393737793 s

# 是不是启动python 本身更加耗时 啊
# 搞错了 time.time() 是s 而不是ms 所以还是有点耗时的 虽然放在 固态了

# load moudle cost 5.962 s
# read file cost 0.577 s

# print("load moudle cost {} ms".format(end))
print("load moudle cost {:.3f} s".format(end-start))
# filename='d:\\py\\data\\Calendar.csv'
# filename='Calendar.csv'
# 感觉python 的库放在 固态 好像也没有很快？
table="US_County_summary_covid19_confirmed_transpose"
filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed_transpose.csv"
# filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed.csv"
# filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed_calendar.csv"
# df=pd.read_csv(filename,parse_dates=['date'])
# df=pd.read_csv(filename,parse_dates=['date'],encoding="gbk")
start=time.time()
df=pd.read_csv(filename,parse_dates=['date'],encoding="utf-8")
end=time.time()
# https://blog.csdn.net/weixin_39705018/article/details/110485262
print("read file cost {:.3f} s".format(end-start))
# 他是只有 date 一列的 ，但是疫情的数据是 有很多列 日期
df.set_index('date', inplace=True)
# value_col_name="confirmed"
# value_col_name="Osage County"
value_col_name="Cass County"

# UnicodeDecodeError: 'utf-8' codec can't decode byte 0x87 in position 10: invalid start byte
# cmap="autumn"
# cmap="RdYlGn"
# cmap="reds"
cmap="RdYlGn_r"
# 这个东西是设置日历图的渐变颜色,但是一开始不知道去哪里找这个参数
# 网上找到的好像也没有合适这张图的,因为他是确诊人数,所以我觉得他从红色变到绿色有点不好
# 网上查到有autumn,试了一下,效果也不是太好
# 最后在写了一个错误的参数的时候,他报错了
# 文档：大作业汇总文档.note
# 链接：http://note.youdao.com/noteshare?id=087e19c2ef61234fecb31c0e6bb78264&sub=WEB3fd1d5ad047fa2567e62438e32d20346
vals=df[value_col_name]
# vals=df.loc[2]
fig,ax=calmap.calendarplot(vals,  fillcolor='grey', 
                           linecolor='w',linewidth=0.1,cmap=cmap,
                           yearlabel_kws={'color':'black', 'fontsize':12},
                            fig_kws=dict(figsize=(10,5),dpi= 80))
fig.colorbar(ax[0].get_children()[1], ax=ax.ravel().tolist())
# fig title 
# https://www.pynote.net/archives/2261
title=table+" "+value_col_name
# fig.title(title)

# plt.title 的位置
# https://www.cnblogs.com/ddfs/p/11798597.html
plt.title(title)
plt.show()
# 一个小细节是 plt.title 放在 plt.show()后面的话 title 就不显示了
# plt.title(title)
# 放在下面是不显示的
#fig.savefig('日历图1.pdf')

