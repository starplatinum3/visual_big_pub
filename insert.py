import pandas as pd
import numpy as np

data = {'a': [4, 6, 5, 7, 8],
        'b': ['w', 't', 'y', 'x', 'z'],
        'c': [1, 0, 6, -5, 3],
        'd': [3, 4, 7, 10, 8],
        }
df = pd.DataFrame(data, index=['one', 'two', 'three', 'four', 'five'])
print(df)

#        a  b  c   d
# one    4  w  1   3
# two    6  t  0   4
# three  5  y  6   7
# four   7  x -5  10
# five   8  z  3   8

new_df = pd.DataFrame({'a': [1, 2, 3, 3, 4],
                       'b': [1, 2, 3, 3, 4],
                       'c': [22, 33, 22, 44, 66],
                       'd': [1, 2, 3, 3, 4]
                       },
                      index=['one', 'two', 'three', 'four', 'five'])
print(new_df)

#        a  b   c  d
# one    1  1  22  1
# two    2  2  33  2
# three  3  3  22  3
# four   3  3  44  3
# five   4  4  66  4

# 取出要插入的行
insertRow = df[1: 2]
# insertRow = df.iloc[2, :]    # 切片操作，行取第二行，列取所有
# insertRow = df.iloc[2]    # 第二行，返回位置索引为1，也就是第二行数据。位置索引，和列表索引类似，里面只能是数字
# insertRow = df.loc['two']    # 返回标签为‘two’的数据
print(insertRow)

#      a  b  c  d
# two  6  t  0  4

newData = new_df.append(insertRow)
print(newData)

#        a  b   c  d
# one    1  1  22  1
# two    2  2  33  2
# three  3  3  22  3
# four   3  3  44  3
# five   4  4  66  4
# two    6  t   0  4

