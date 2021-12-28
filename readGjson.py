
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
filepath=r"D:\proj\visualization\bigwork\input\China_admin1.geojson"
import json
# 读取json文件内容,返回字典格式
with open(filepath,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
    # print('这是文件中的json数据：',json_data)
    print('这是读取到文件数据的数据类型：', type(json_data))
features=json_data['features']

for f in features:
    properties=f["properties"]
    geometry=f["geometry"]
    coordinates=geometry["coordinates"]
    print("coordinates")
    print(coordinates)
