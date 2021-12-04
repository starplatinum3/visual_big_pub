
class Record:
    def __init__(self,date,place,num):
        self.date=date
        self.place=place
        self.num=num
    # def __str__(self):


        
import pandas as pd
import json 

def to_json1(df,orient='split'):
    return df.to_json(orient = orient, force_ascii = False)
    
def to_json2(df,orient='split'):
    df_json = df.to_json(orient = orient, force_ascii = False)
    return json.loads(df_json)

state_filename_base=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\State_level_summary\US_State_summary_covid19_{}_trpo.xlsx"

# df_confirmed = pd.read_excel(state_filename_base.format("confirmed"),header=None)
fullname=state_filename_base.format("confirmed")
df_confirmed = pd.read_excel(state_filename_base.format("confirmed"))
# json1 = to_json1(df)
# json2 = to_json2(df)
# ————————————————
# 版权声明：本文为CSDN博主「風の唄を聴け」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_42902669/article/details/90293004
# df_confirmed_json=to_json1(df_confirmed)
df_confirmed_json=to_json2(df_confirmed)


datas=df_confirmed_json["data"]
# print("datas")
# print(datas)
idx=0
cols=df_confirmed_json["columns"]
# cols
# map[col]=[1,1,1,1,]
# for i in datas:
#     # print(i)
#     if( idx<=4):
#         print(i)
#     idx+=1
#     # if( idx>=4):

def parse_row(row,cols):
    idx=0
    records=[]
    date=row[0]
    # print("date")
    # print(date)
    # print("row")
    # print(row)
    for val in row:
        if idx==0:
            idx+=1
            continue
        num=val
        # print("val")
        # print(val)
        rec=Record(date,cols[idx],num)
        records.append(rec)
        idx+=1
    # print("records")
    # print(records)
    return records

def append_all(origin_lst:list,food_lst):
    for i in food_lst:
        origin_lst.append(i)

all_recs=[]
for row in datas:
    # print(i)
    row_recs=parse_row(row,cols)
    # all_recs.appendA
    # appendAll python
    append_all(all_recs,row_recs)
    # if( idx<=4):
    #     print(row)
    # idx+=1
    # if( idx>=4):
# dateframe json
# data_json={"date":[],"confirmed":[],"couty":[]}




import datetime



def time_s_to_date_str(timeStampMs):
    timeStampMs/=1000.0
    dateArray = datetime.datetime.utcfromtimestamp(timeStampMs)

    otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
    return otherStyleTime


date_lst=[]
confirmed_lst=[]
couty_lst=[]
rec:Record
for rec in all_recs:
    # date_lst.append(rec.date)
    int_date=int(rec.date)
    time_str=time_s_to_date_str(int_date)
    # pd.datetime
    # time1=pd.to_datetime(rec.date)
    # time_str=time_s_to_date_str(rec.date)
    date_lst.append(time_str)
    # date_lst.append(time1)
    confirmed_lst.append(rec.num)
    couty_lst.append(rec.place)

data_json={"date":date_lst,"confirmed":confirmed_lst,"couty":couty_lst}

# timeStamp = 1381419600

# dateArray = datetime.datetime.utcfromtimestamp(timeStamp)

# otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
# print("otherStyletime")
# print(otherStyletime)
# otherStyletime == "2013-10-10 23:40:00"


# df
# 1579651200000
# 时间戳 转化 时间 python
# 时间戳 转化 pd datetime

# timeStamp=1579651200000
# str_time=time_s_to_date_str(timeStamp)
# print("str_time")
# print(str_time)
# python ms 转化为 日期

pd=pd.DataFrame(data_json)
print("pd")
print(pd)
# 使用index=False
pd.to_excel("db_kind/"+fullname.split("\\")[-1],index=False)
# pd.to_excel("db_kind/"+fullname.split("\\")[-1])
# to_excel 不要 第一列