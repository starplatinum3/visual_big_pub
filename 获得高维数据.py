
import  os
import pandas as pd
# 因为数据集给的数据都是一天天的 而且只有一个特定的属性 比如降水量
# 这样不是很合适数据分析 所以我写了一个代码 用来合并这些数据为一个表格
# 过程也挺艰辛的 花了一上午 主要就是df 的 append ,会返回一个df
# 要把这个df 重新复制 ,他不是替换的
root_dir=r"D:\visual\Environmental factors\China_admin1"
# dir_path=r"D:\visual\Environmental factors\China_admin1\Humidity"
# dir_path=r"D:\visual\Environmental factors\China_admin1\Precipitation"
# lst_dir=os.listdir(dir_path)
def read_one_factor(factor:str):
    abs_factor_path=os.path.join(root_dir,factor)
    print("abs_factor_path")
    print(abs_factor_path)
    lst_dir=os.listdir(abs_factor_path)
    tot_df=pd.DataFrame()
    for f in lst_dir:
        date=f.split("_")[0]
        # abs_path=os.path.join(dir_path,f)
        abs_path=os.path.join(abs_factor_path,f)
        # print("abs_path")
        # print(abs_path)
        pd.read_csv(abs_path)
        # tot_df=read_one_file(abs_path,date,tot_df)
        tot_df=read_one_file(abs_path,date,tot_df,factor)
    return tot_df

def all_factors():
    tot_df=pd.DataFrame()
    factor_lst=['Humidity','Precipitation','Temper_T2MMEAN']
    # df 根据index join
    df_lst=[]
    for fac in factor_lst:
        fac_df=read_one_factor(fac)
        # print("fac_df")
        # print(fac_df)
        # print("tot_df")
        # print(tot_df)
        print("fac_df")
        print(fac_df)
        df_lst.append(fac_df)
        # tot_df=tot_df.join(fac_df)
        # tot_df=tot_df.append(read_one_factor)
    # print("tot_df")
    # print(tot_df)
    fir=True
    idx=0
    for df in df_lst:
        if idx==0:
            tot_df["HASC_1"]=df["HASC_1"]
            tot_df["Humidity"]=df["Humidity"]
            tot_df["date"]=df["date"]
            # fir=False
        else:
            fac_str=factor_lst[idx]
            tot_df[fac_str]=df[fac_str]
        idx+=1
    print("tot_df")
    print(tot_df)
    # tot_df.to_csv("all_factors_china.csv")
# tot_df=pd.DataFrame({"HASC_1":None,"Humidity":None,"date":None})
# tot_df=pd.DataFrame({"HASC_1":None,"Humidity":None,"date":None},index=[0])
# tot_df=pd.DataFrame({"HASC_1":[],"Humidity":[],"date":[]},index=[0])
# tot_df=pd.DataFrame({"HASC_1":[],"Humidity":[],"date":[]})
# tot_df=pd.DataFrame({"HASC_1": "CN.AH"    ,"Humidity":0.020212,"date":"20210731"})
# tot_df=pd.DataFrame({"HASC_1": "CN.AH"    ,"Humidity":0.020212,"date":"20210731"},index=[0])
# tot_df=pd.DataFrame()
# print("tot_df")
# print(tot_df)

# ValueError: Empty data passed with indices specified.

# def
# a = []
# for line in insert_lines:
#     line = dict(line)
#     a.append(line)
# a = pd.Dataframe(a)

def read_one_file(abs_path,date,tot_df,factor):
    df=pd.read_csv(abs_path)
    df["date"]=date
    # df["Humidity"]=df["Mean"]
    df[factor]=df["Mean"]
    # df["Mean"]=
    # print("df")
    # print(df)
    # df append
    # ValueError: If using all scalar values, you must pass an index
    # add_df=df[["HASC_1","Humidity","date"]]
    # https://blog.csdn.net/qq_36387683/article/details/86016913
    # print("add_df")
    # print(add_df)
    # tot_df.append(df[["HASC_1","Humidity","date"]])
    # tot_df=tot_df.append(df[["HASC_1","Humidity","date"]],ignore_index=True)
    tot_df=tot_df.append(df[["HASC_1",factor,"date"]],ignore_index=True)
    # https://blog.csdn.net/weixin_39750084/article/details/81429037
    # tot_df.concat(df[["HASC_1","Humidity","date"]])
    # tot_df.concat
    # pd.concat(tot_df,add_df)
    # add_df.data
    # for line in add_df:
    # 把一个 df 插入到另外一个 df 下面
    # for line in add_df.data:
    #     # tot_df.append(line)
    #     print("line")
    #     print(line)
    #     tot_df.append(dict(line))
    # pd.concat([tot_df,add_df], axis=0)
    # pd.concat([tot_df,add_df], axis=1)
    # print("tot_df")
    # print(tot_df)
    # pandas 添加行
    return tot_df

all_factors()

# for f in lst_dir:
#     date=f.split("_")[0]
#     abs_path=os.path.join(dir_path,f)
#     pd.read_csv(abs_path)
#     tot_df=read_one_file(abs_path,date,tot_df)
#
# print("tot_df")
# print(tot_df)
#
# df=pd.DataFrame({})
#
# # 取出要插入的行
# insertRow = df[1: 2]
# # insertRow = df.iloc[2, :]    # 切片操作，行取第二行，列取所有
# # insertRow = df.iloc[2]    # 第二行，返回位置索引为1，也就是第二行数据。位置索引，和列表索引类似，里面只能是数字
# # insertRow = df.loc['two']    # 返回标签为‘two’的数据
# print(insertRow)
#
# #      a  b  c  d
# # two  6  t  0  4
#
# newData = new_df.append(insertRow)
# print(newData)

#        a  b   c  d
# one    1  1  22  1
# two    2  2  33  2
# three  3  3  22  3
# four   3  3  44  3
# five   4  4  66  4
# two    6  t   0  4
