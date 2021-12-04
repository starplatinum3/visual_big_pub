
# 引入模块
import time, datetime

# 使用time
# timeStamp = 1381419600
timeStamp = 1579651200000/1000
timeArray = time.localtime(timeStamp)
otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
print(otherStyleTime)   # 2013--10--10 23:40:00
# 使用datetime
timeStamp = 1381419600
dateArray = datetime.datetime.fromtimestamp(timeStamp)
otherStyleTime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
print(otherStyleTime)   # 2013--10--10 23:40:00
# 使用datetime，指定utc时间，相差8小时
timeStamp = 1381419600
dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
otherStyleTime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
print(otherStyleTime)   # 2013--10--10 15:40:00


#!-*- coding:UTF-8 -*-
'''
Created on 2015-4-14
'''
import datetime
import time

timeStamp = 1427349630000
timeStamp /= 1000.0
print (timeStamp)
timearr = time.localtime(timeStamp)
otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timearr)
print (otherStyleTime)