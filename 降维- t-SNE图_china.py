# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from plotnine import *
from sklearn import manifold, datasets
# 因为要做大作业 所以把老师的代码也看了看 试试能不能用在大作业上
# 做了写注释 和想法 感觉自己算是在 花时间 和尝试理解 了吧。。
#------------------------------------(a) 四维数据的iris数据集----------------------------------------------------------
iris = datasets.load_iris()

filename="all_factors_china.csv"
df=pd.read_csv(filename)
df=df[["Humidity","Precipitation","Temper_T2MMEAN"]]
# 修改了颜色
DeepPink="#FF1493"
BlueViolet="#8A2BE2"
SkyBlue="#87CEEB"
Gold="#FFD700"
black="black"
color3="#FC4E07"


# shape='o'
# colour='k'
# 这个意思是按照原来的颜色吗
#  figure_size = (5,5),
# aspect_ratio =1
#  dpi = 100
# n_components=2, 
#  random_state=501


# 修改了一些参数
# shape="+"
# 貌似是形状的问题 ，+ 是 没有颜色分别的 只能是 o
shape="o"
# colour=SkyBlue
colour='k'
# 这个意思是按照原来的颜色吗
# 这波颜色还不敢乱改，改了 就只有一个颜色了
# 改成k 也没用啊
legend_text_color=BlueViolet
# figure_size=(4,6)
figure_size = (5,5)
# aspect_ratio =4
aspect_ratio =1
# dpi = 200
# 这个设置貌似让字变得很大 都看不见了有些
dpi = 100
# n_components=4
# n_components=3
n_components=2
# 怀疑 我是不是要画个 3维的图像才能显示 n_components=3 的 分类效果
# https://blog.csdn.net/qq_23534759/article/details/80457557
# random_state：int或RandomState实例或None（默认）
# 伪随机数发生器种子控制。如果没有，请使用numpy.random单例。请注意，不同的初始化可能会导致成本函数的不同局部最小值。
# 貌似是说 n_components 必须 <4  所以就换了个3
# ValueError: 'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree.
# https://stackoverflow.com/questions/66592804/t-sne-can-not-convert-high-dimension-data-to-more-than-4-dimension
# random_state=600
random_state=200
# 这个参数的修改 肉眼没有看出大区别 可能是改的方向不对

# Manifold日语翻译为多样体，这是比较符合原意的。
# t-SNE(TSNE)将数据点之间的相似度转换为概率。
# https://www.cnblogs.com/bonelee/p/7849867.html
# 以t-SNE为例子，代码如下，n_components设置为3，也就是将64维降到3维，i
# 大概就是这个图 分的越开 就是越好吧 分类的任务
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=random_state)
# print("iris.data")
# print(iris.data)
# https://www.cnblogs.com/moonlightml/p/9795521.html
# https://www.cnblogs.com/Belter/p/8831216.html
# X_tsne = tsne.fit_transform(iris.data)
X_tsne = tsne.fit_transform(df)
# 四个 参数 是4维度的意思吗
# 肺炎是什么维度呢？ 每个日期算个维度吗？ 应该不是这个概念吧
# 于是不是很懂 降维能在 大作业上用到吗
# https://xueshu.baidu.com/usercenter/paper/show?paperid=1a6e0vw0ep6502n0u0430gw0bb228862
# 话说肺炎算是个 分类任务吗，硬要说也可以吧 根据他的地域特征 来推测他是不是肺炎
# 如果是美国纽约的话 可能性就很大了 虽然也有点道理，但是貌似没有更多的特征 ，但是感觉硬套模型也是可以的

print("X_tsne")
print(X_tsne)
# 明确的  Categorical Categorical类别的;
# 山鸢尾（学名：Iris setosa Pall. ex Link）
# iris.target_names
# 鸢尾属植物;
# 
# ['setosa' 'versicolor' 'virginica']
# diervilla versicolor 路边花
# 大家都知道这个鸢尾花样本数据集一共分成3个Species(品种):setosa、versicolor、和virginica。
# https://www.sohu.com/a/291002020_354986
# iris 应该是个 默认数据集
print("iris.target_names")
print(iris.target_names)
print("iris.target")
print(iris.target)
# 要这个的话 首先要贴标签
# 那么肺炎的 人数的话 标签是？
# 确诊之类的就是标签了
# https://www.cnblogs.com/cgmcoding/p/13603370.html
target=pd.Categorical.from_codes(iris.target,iris.target_names)
# 就是给这些target 变成名字( 文字）吗
# 这块 DistributedY1   DistributedY2
# 两个维度 是不是意味着 n_components 必然是2  我也不是很理解
# 虽然 n_components =3 貌似X_tsne 出来也是 2维的
print("X_tsne.shape")
print(X_tsne.shape)
# X_tsne.shape
# (150, 3)
# 确实是3维度的
# 但是标签是?
df=pd.DataFrame(dict(DistributedY1=X_tsne[:, 0],DistributedY2=X_tsne[:, 1],target=target))
print("df")
print(df)
base_plot2=(ggplot(df, aes('DistributedY1','DistributedY2',fill='target')) +
   geom_point (alpha=1,size=3,shape=shape,colour=colour)+
  # 绘制透明度为0.2 的散点图
 # stat_ellipse( geom="polygon", level=0.95, alpha=0.2) +
  #绘制椭圆标定不同类别.
  #scale_color_manual(values=("#00AFBB","#FC4E07")) +#使用不同颜色标定不同数据类别
#   scale_fill_manual(values=("#00AFBB", "#E7B800", "#FC4E07"),name='group')+  #使用不同颜色标定不同椭类别
  scale_fill_manual(values=(Gold, BlueViolet, "#FC4E07"),name='group')+  #使用不同颜色标定不同椭类别
  theme(
       #text=element_text(size=15,face="plain",color="black"),
       axis_title=element_text(size=13,face="plain",color="black"),
       axis_text = element_text(size=12,face="plain",color="black"),
       legend_text = element_text(size=11,face="plain",color=legend_text_color),
       legend_background=element_blank(),
       legend_position=(0.3,0.25),
       aspect_ratio =aspect_ratio,
       figure_size =figure_size,
       dpi = dpi
       )
)
print(base_plot2)

#----------------------------------------(b) 93维数据的train数据集--------------------------------------------------
def dimen93():
    df=pd.read_csv('Tsne_Data.csv')
    df=df.set_index('id')

    num_rows_sample=5000
    # 这个挺花时间的 因为是 93维吗。。。

    # 修改了参数
    s = 0.6
    l = 0.6
    h=0.6
    color_space='husl'
    # color_space='xvYCC'
    # https://www.colorspace.com.cn/kb/2018/07/03/%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4color-space/

    df = df.sample(n=num_rows_sample)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(df.iloc[:,:-1])

    df=pd.DataFrame(dict(DistributedY1=X_tsne[:, 0],DistributedY2=X_tsne[:, 1],target=df.iloc[:,-1]))

    base_plot2=(ggplot(df, aes('DistributedY1','DistributedY2',fill='target')) +
    geom_point (alpha=1,size=2,shape='o',colour='k',stroke=0.1)+
    # 绘制透明度为0.2 的散点图
    # stat_ellipse( geom="polygon", level=0.95, alpha=0.2) 
    #scale_color_manual(values=("#00AFBB","#FC4E07")) +#使用不同颜色标定不同数据类别
    # scale_fill_cmap(name ='Set1')+
    scale_fill_hue(s = s, l = l, h=h,color_space=color_space)+
    #   xvYCC
        # scale_fill_hue(s = 0.99, l = 0.65, h=0.0417,color_space='husl')+
    xlim(-100,100)+
    theme(
        #text=element_text(size=15,face="plain",color="black"),
        axis_title=element_text(size=13,face="plain",color="black"),
        axis_text = element_text(size=12,face="plain",color=DeepPink),
        legend_text = element_text(size=11,face="plain",color="black"),
        aspect_ratio =aspect_ratio,
        figure_size = figure_size,
        dpi = dpi
        )
    )
    print(base_plot2)