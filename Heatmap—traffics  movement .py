import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import warnings
warnings.filterwarnings('ignore')
import folium
import folium.plugins as plugins
from folium.plugins import HeatMap


filepath=r"D:\proj\visualization\bigwork\input\China_admin1.geojson"
import json
# 读取json文件内容,返回字典格式
with open(filepath,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
    # print('这是文件中的json数据：',json_data)
    print('这是读取到文件数据的数据类型：', type(json_data))
features=json_data['features']
features0=features[0]
geometry0=features0["geometry"]
coordinates0=geometry0["coordinates"]
# data=coordinates0[0]
data=coordinates0[0][0]
# print("coordinates0")
# print(coordinates0)
# 里面是所有的 轮廓的 经纬度

# coordinates01=coordinates0[0][1]
# print("coordinates01")
# print(coordinates01)
# IndexError: list index out of range
# 后面没有 数据了 也就是说 这个文件就是个经纬度轮廓数据

# for f in features:
#     # properties=f["properties"]
#     geometry=f["geometry"]
#     coordinates=geometry["coordinates"]
#     print("coordinates")
#     print(coordinates)


# save_pic=False
save_pic=True
# --------------------------define the world map-----------------------
world_map = folium.Map()
# display world map

world_map

# ----------------------------Heatmap movement Example-------------------------
# 
# - Let us use HeatMap plugin to generate an animation in Folium

#Define a random seed
np.random.seed(3141592)
Number=50
#Generate random data around latitude = 23 (Tropic of Cancer) longitude of 77
# initial_data = (
#     np.random.normal(size=(Number, 2)) * np.array([[1, 1]]) +
#     np.array([[23, 77]])
# )
#Generate Heatmap movement dataset
# move_data = np.random.normal(size=(Number, 2)) * 0.02
# data = [(initial_data + move_data * i).tolist() for i in range(Number)]
# weight = 1  # default value
# for time_entry in data:
#     for row in time_entry:
#         row.append(weight)
print("data")
print(data)
out_data=[]

for i in data:
    out_data.append([i[1],i[0]])
    # lat_sum+=i[1]
    # lot_sum+=i[0]
# print("data.shape")
# print(data.shape)
#Define the Folium Map
# hmwtMap = folium.Map([23., 77.], tiles='OpenStreetMap', zoom_start=4)
# #Add the Heatmap data to HeatMapWithTime plugin        
# hmwtPlugin = plugins.HeatMapWithTime(data)
# hmwtPlugin.add_to(hmwtMap)
# #Render the Finished Map
# # if save_pic:
# #     SimpleHmDataMap.save("Simple Example_mqp_77.html")
# hmwtMap
# out_data=[lat_sum,lot_sum]

def put_sum_lot():
    lat_sum=0
    lot_sum=0
    for i in out_data:
        lat_sum+=i[0]
        lot_sum+=i[1]
        # out_data=[lat_sum,lot_sum]
    sum_data=[[lat_sum,lot_sum]]
    return sum_data
#------------------------------------Heatmap----------------------
# data = (
#     np.random.normal(size=(100, 3)) *
#     np.array([[1, 1, 1]]) +
#     np.array([[28, 77, 1]])
# ).tolist()
# print("data")
# print(data)
# SimpleHmDataMap = folium.Map([28., 77.], tiles='OpenStreetMap', zoom_start=6)
# SimpleHmDataMap = folium.Map([28., 77.], tiles='OpenStreetMap', zoom_start=20)
# SimpleHmDataMap = folium.Map([28., 100.], tiles='OpenStreetMap', zoom_start=6)
SimpleHmDataMap = folium.Map([28., 77.], tiles='OpenStreetMap', zoom_start=6)
# Map([28., 77.] 一开始 的经纬度吧
# HeatMap(data).add_to(SimpleHmDataMap)
HeatMap(out_data).add_to(SimpleHmDataMap)

# sum_data=put_sum_lot()
# HeatMap(sum_data).add_to(SimpleHmDataMap)
# SimpleHmDataMap.save("Simple Example.html")
# ', zoom_start=20) 一开始就放的很大
# SimpleHmDataMap.save("Simple Example_zoom_start20.html")
if save_pic:
    pass
    SimpleHmDataMap.save("Simple Example_mqp_77.html")
    # SimpleHmDataMap.save("Simple 安徽.html")
# SimpleHmDataMap