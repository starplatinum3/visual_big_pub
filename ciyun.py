# import jieba

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

import wordcloud #导入词云库
# https://blog.csdn.net/moshanghuali/article/details/84667136
# SIMYOUTTF_downcc
plt.rcParams['font.sans-serif']=[u'SimHei']
filename=r"D:\download\GIS疫情地图2020全年-至今数据\【GIS点滴疫情地图·2020年01月02日-2021年01月25日】国内每天疫情统计.xlsx"
# df:pd
df:pd.DataFrame
df=pd.read_excel(filename)
# df.hist
# df.
# 从 0 开始
# suspect0=df["suspect"][0]
suspect0=df["suspect"][11]
# 的所有的 数据 我们要他的第一行
print("suspect0")
print(suspect0)

len_df=len(df)
print("len_df")
print(len_df)
df["seriousCount"]=df["seriousCount"].fillna(0)
# "SIMYOUTTF_downcc/SIMYOU.TTF" ,
# 因为nan会 不能转化 int  需要变成0
df["currentdiagnose"]=df["currentdiagnose"].fillna(0)
WC = wordcloud.WordCloud(font_path ="SIMYOU.TTF" ,
                         max_words=100,height= 800,width=800,background_color='white',repeat=False,mode='RGBA')
#设置词云图对象属性
# con = WC.generate_from_frequencies(dic_frec)
# # https://www.cnblogs.com/liangmingshen/p/11312257.html
# plt.imshow(con)
# plt.axis("off")
# # https://blog.csdn.net/moshanghuali/article/details/84667136
# # 词云图 都是 方框
# plt.show()

def plt_row(row_num:int):

    # dic_frec={"suspect":df["suspect"][row_num],
    #           "cure":df["cure"][row_num],
    #           "death":df["death"][row_num],
    #           "seriousCount":df["seriousCount"][row_num],
    #           }
    dic_frec={"疑似":df["suspect"][row_num],
              "治愈":df["cure"][row_num],
              "死亡":df["death"][row_num],
              "严重":df["seriousCount"][row_num],
              "确诊":df["diagnose"][row_num],
              "现在确诊":df["currentdiagnose"][row_num],

              }

    try:
        con = WC.generate_from_frequencies(dic_frec)
    except ValueError as e:
        print("dic_frec")
        print(dic_frec)
    # https://blog.csdn.net/clksjx/article/details/105720120
    # https://www.cnblogs.com/liangmingshen/p/11312257.html
    # im=plt.imshow(con)
    plt.axis("off")
    plt.title(f"时间 {df['day'][row_num]}")
    return plt.imshow(con)
    # plt.axis("off")
    # # https://blog.csdn.net/moshanghuali/article/details/84667136
    # # 词云图 都是 方框
    #
    #
    # return  plt.show()
    # plt 多次画图 转化为gif

# fig = plt.figure()
# plt.ion()
# # https://blog.csdn.net/weixin_42990464/article/details/112347386
# plt.show()
# ims = []
# for i in range(1,10):
#     im = plt.plot(np.linspace(0, i,10), np.linspace(0, np.random.randint(i),10))
#     ims.append(im)
# plt.draw()
# plt.pause(0.2)



from celluloid import Camera


fig = plt.figure()
camera = Camera(fig)
# fig = plt.figure()
plt.ion()
# https://blog.csdn.net/weixin_42990464/article/details/112347386
# plt.show()
ims = []
print("start")
for row_num in range(len_df):
    im=plt_row(row_num)
    ims.append(im)
    # print(im)
    camera.snap()
print("ims")
print(ims)
plt.draw()
plt.pause(0.2)

# for i in range(10):
#     plt.plot([i] * 10)
#     camera.snap()
animation = camera.animate()
animation.save('词云图动态.gif')


# plt.hist(x, y,label="确诊")
# plt.hist(x, df["suspect"],label="疑似")
# plt.hist(x, df["cure"],label="治愈")
# plt.hist(x, df["death"],label="死亡")
# plt.hist(x, df["seriousCount"],label="严重")
# plt.hist(x, df["currentdiagnose"],label="目前确诊")
#

# data=pd.read_csv("动漫评价数据集/anime_t.csv",encoding="gbk")
# data=pd.read_csv("bangumi_csv/bangumi_1.csv",encoding="gbk")
# seg_list = jieba.cut("我来到北京清华大学", cut_all = True)
# print("Full Mode:", ' '.join(seg_list))
def join_all_data():
    tot_num=49
    datas=[]
    for num in range(1,tot_num+1):
        try:
            datas.append(pd.read_csv("bangumi_csv/bangumi_{}.csv".format(num),encoding="gbk"))
        except:
            datas.append(pd.read_csv("bangumi_csv/bangumi_{}.csv".format(num),encoding="utf-8"))
    # data=datas[0]
    data= pd.DataFrame()
    for num in range(tot_num):
        data=data.append(datas[num],ignore_index=True)

    return data

def push_list(lst,creators):
    creators=creators.split("/")
    for crea in creators:
        creator=crea.strip()
        if creator=="创作者不明":
            continue
        lst.append(creator)
        
        
def split_creators(data):
    creators_list=[]
    for creators in data["创作者"]:
        push_list(creators_list,creators)
    return creators_list

def list_to_blank_words(lst):
    res=""
    for w in lst:
        if w=="创作者不明" or w=="連載":
            continue
        res+=" "+w
    return res

def gen_dic_frec(lst):
    dic_frec={}
    for sth in lst:
        if sth not in dic_frec:
            dic_frec[sth]=1
        else:
            dic_frec[sth]+=1

    return dic_frec


def plt_wordcloud():
    data=join_all_data()
    # r"D:\project\jsProject\animals\fonts\fontawesome-webfont.ttf"
    WC = wordcloud.WordCloud(font_path ="SIMYOUTTF_downcc/SIMYOU.TTF" ,
                             max_words=100,height= 800,width=800,background_color='white',repeat=False,mode='RGBA')
    #设置词云图对象属性
    # https://blog.csdn.net/weixin_42240667/article/details/104955036

    # ————————————————
    # 版权声明：本文为CSDN博主「moshanghuali」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/moshanghuali/article/details/84667136
    # conten = ' '.join(jieba.lcut(st1)) #此处分词之间要有空格隔开，联想到英文书写方式，每个单词之间都有一个空格。
    creators=split_creators(data)
    # 这是一个列表

    # print(creators)
    # words_list=list_to_blank_words(creators)
    # print(words_list)

    plt.rcParams['font.sans-serif']=[u'SimHei']

    # conten = words_list
    # con = WC.generate(conten)
    dic_frec=gen_dic_frec(creators)
    # print(dic_frec)


    con = WC.generate_from_frequencies(dic_frec)
    # https://www.cnblogs.com/liangmingshen/p/11312257.html
    plt.imshow(con)
    plt.axis("off")
    # https://blog.csdn.net/moshanghuali/article/details/84667136
    # 词云图 都是 方框
    plt.show()