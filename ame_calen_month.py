# -*- coding: utf-8 -*-
from os import pardir
import pandas as pd
import numpy as np
from plotnine import *

import matplotlib.pyplot as plt 

from datetime import datetime

# filename='d:\\py\\data\\Calendar.csv'
# filename='Calendar.csv'
csv_name="US_County_summary_covid19_confirmed_transpose"
filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed_transpose.csv"
df=pd.read_csv(filename,parse_dates=['date'])

df['year']=[d.year for d in df['date']]
# print("df['year']")
# print(df['year'])
# 这个是有的

# df=df[df['year']==2017]
# df=df[df['year']==2016]
year=2021
# year=2020
df=df[df['year']==year]
# 因为没有 2016 的数据。。。
df['month']=[d.month for d in df['date']]
month_label=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
df['monthf']=df['month'].replace(np.arange(1,13,1), month_label)

df['monthf']=df['monthf'].astype(pd.CategoricalDtype(categories=month_label,ordered=True))

df['week']=[int(d.strftime('%W')) for d in df['date']]

df['weekay']=[int(d.strftime('%u')) for d in df['date']]

week_label=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
df['weekdayf']=df['weekay'].replace(np.arange(1,8,1), week_label)
df['weekdayf']=df['weekdayf'].astype(pd.CategoricalDtype(categories=week_label,ordered=True))


df['day']=[d.strftime('%d') for d in df['date']]
# 应该是 day 没有数据的问题
# print("df['day']")
# print(df['day'])
# df['day']
# Series([], Name: day, dtype: float64)

df['monthweek']=df.groupby('monthf')['week'].apply(lambda x: x-x.min()+1)

fill="Osborne County"
# fill='value'
# https://blog.csdn.net/qq_42458954/article/details/82356061
# labs : labs(x = "这是 X 轴", y = "这是 Y 轴", title = "这是标题") ## 修改文字
title=csv_name+"_{}_year_{}".format(fill,year)
base_plot=(ggplot(df, aes('weekdayf', 'monthweek', fill=fill)) + 
  geom_tile(colour = "white",size=0.1) + 
  scale_fill_cmap(name ='Spectral_r')+
  geom_text(aes(label='day'),size=8)+
  facet_wrap('~monthf' ,nrow=3) +
   labs(title =title)+
  scale_y_reverse()+
  xlab("Day") + ylab("Week of the month") +
  theme(strip_text = element_text(size=11,face="plain",color="black"),
        axis_title=element_text(size=10,face="plain",color="black"),
         axis_text = element_text(size=8,face="plain",color="black"),
         legend_position = 'right',
         legend_background = element_blank(),
         aspect_ratio =0.85,
        figure_size = (8, 8),
        dpi = 100
  ))

#   plotnine.exceptions.PlotnineError: 'Faceting variables must have at least one value'
  
print(base_plot)






