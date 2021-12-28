'''
Codes for the paper 'A Time-dependent SIR model for COVID-19 with Undetectable Infected Persons'

Authors: Yi-Cheng Chen, Ping-En Lu, Cheng-Shang Chang, and Tzu-Hsuan Liu
Institute of Communications Engineering, National Tsing Hua University, Hsinchu 30013, Taiwan, R.O.C.
Email: j94223@gmail.com

The latest version of the paper will be placed on this link: http://gibbs1.ee.nthu.edu.tw/A_TIME_DEPENDENT_SIR_MODEL_FOR_COVID_19.PDF

We have uploaded the paper to arXiv (https://arxiv.org/abs/2003.00122).

CVID2019冠状病毒疾病的时间依赖性SIR模型的代码
作者：陈一诚、吕炳恩、程尚昌、刘子玄
国立清华大学通信工程研究所，台湾新竹30013。
电邮：j94223@gmail.com
最新版本的论文将放在此链接上：http://gibbs1.ee.nthu.edu.tw/A_TIME_DEPENDENT_SIR_MODEL_FOR_COVID_19.PDF
我们已将论文上传至arXiv(https://arxiv.org/abs/2003.00122).

'''
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import pandas as pd


# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def data_spilt(data, orders, start):
    x_train = np.empty((len(data) - start - orders, orders))
    y_train = data[start + orders:]

    for i in range(len(data) - start - orders):
        x_train[i] = data[i + start:start + orders + i]

    # Exclude the day (Feb. 12, 2020) of the change of the definition of confirmed cases in Hubei China.
    x_train = np.delete(x_train, np.s_[28 - (orders + 1) - start:28 - start], 0)
    y_train = np.delete(y_train, np.s_[28 - (orders + 1) - start:28 - start])

    return x_train, y_train


def ridge(x, y):
    print('\nStart searching good parameters for the task...')
    parameters = {'alpha': np.arange(0, 0.100005, 0.000005).tolist(),
                  "tol": [1e-8],
                  'fit_intercept': [True, False],
                  'normalize': [True, False]}

    clf = GridSearchCV(Ridge(), parameters, n_jobs=-1, cv=5)
    clf.fit(x, y)

    print('\nResults for the parameters grid search:')
    print('Model:', clf.best_estimator_)
    print('Score:', clf.best_score_)

    return clf


########## data ##########
# data collected from https://voice.baidu.com/act/newpneumonia/newpneumonia
# X_cml = cumulative confirmed cases

# X_cml = np.array([41, 45, 62, 121, 199, 291, 440, 574, 835, 1279, 1985, 2761, 4535, 5997, 7736, 9720, 11821, 14411, 17238, 20471, 24363, 28060, 31211, 34598, 37251, 40235, 42708, 44730, 59882, 63932, 66576, 68584, 70635, 72528, 74279, 75101, 75993, 76392, 77041, 77262, 77779, 78190, 78630, 78959, 79389, 79968, 80174, 80302, 80422, 80565, 80710, 80813, 80859, 80904, 80924, 80955, 80980, 81003, 81201, 81048, 81077, 81116, 81151, 81235, 81300, 81416, 81498, 81600, 81747, 81846, 81960, 82078, 82213, 82341, 82447, 82545, 82631, 82724, 82802, 82875, 82930, 83005, 83071, 83157, 83249], dtype=np.float64)[:-27]
# # recovered = cumulative recovered cases
# recovered = np.array([12, 12, 16, 21, 25, 25, 28, 28, 34, 38, 49, 51, 60, 103, 124, 171, 243, 328, 475, 632, 892, 1153, 1540, 2050, 2651, 3283, 3998, 4742, 5915, 6728, 8101, 9425, 10853, 12561, 14387, 16170, 18279, 20673, 22907, 24757, 27353, 29775, 32531, 36157, 39049, 41675, 44518, 47260, 49914, 52109, 53793, 55477, 57143, 58684, 59982, 61567, 62887, 64216, 65649, 67022, 67863, 68799, 69725, 70547, 71284, 71876, 72382, 72841, 73299, 73791, 74196, 74737, 75122, 75600, 75937, 76225, 76415, 76610, 76785, 76984, 77210, 77348, 77450, 77586, 77711], dtype=np.float64)[:-27]
# # death = cumulative deaths
# # 累计的 cumulative
# death = np.array([2, 3, 3, 3, 4, 6, 9, 18, 25, 41, 56, 80, 106, 132, 170, 213, 259, 304, 361, 425, 491, 564, 637, 723, 812, 909, 1017, 1114, 1368, 1381, 1524, 1666, 1772, 1870, 2006, 2121, 2239, 2348, 2445, 2595, 2666, 2718, 2747, 2791, 2838, 2873, 2915, 2946, 2984, 3015, 3045, 3073, 3100, 3123, 3140, 3162, 3173, 3180, 3194, 3204, 3218, 3231, 3242, 3250, 3253, 3261, 3267, 3276, 3283, 3287, 3293, 3298, 3301, 3306, 3311, 3314, 3321, 3327, 3331, 3335, 3338, 3340, 3340, 3342, 3344], dtype=np.float64)[:-27]
# 不过其实有做工作,比如说他的数据是 np.array 传入的, 写死在内存里,我拿自己的xlsx 里面的数据读入的时候,使用的是一列数据,
# 比如 df["diagnose"] 是 确诊的人数的一个列 ,于是他在进行 (X[1:] - X[:-1] + R[1:] - R[:-1])  运算的时候
# 就会出错, 原因是 他是df ,所以他会错开相减,最后会多出一个数字,举个例子,就是如此
filename=r"D:\download\GIS疫情地图2020全年-至今数据\【GIS点滴疫情地图·2020年01月02日-2021年01月25日】国内每天疫情统计.xlsx"
df = pd.read_excel(filename)
X_cml=np.array(df["diagnose"])
recovered=np.array(df["cure"])
death=np.array(df["death"])


# X_cml=df["diagnose"]
# recovered=df["cure"]
# death=df["death"]



population = 1439323776
########## data preprocess ##########
X = X_cml - recovered - death
R = recovered + death
print("X.shape")
print(X.shape)

print("R.shape")
print(R.shape)

n = np.array([population] * len(X), dtype=np.float64)

print("n.shape")
print(n.shape)

# X.shape
# (58,)
# R.shape
# (58,)
# from sklearn.pipeline import make_pipeline
# n.shape
# (58,)
# (X[1:] - X[:-1] + R[1:] - R[:-1]) .shape
# (57,)
# (X[:-1] * (n[:-1] - X[:-1] - R[:-1])).shape
#
# (57,)

# If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:
# X[1:] .shape
#
# (57,)
# kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}
# X[:-1].shape
# model.fit(X, y, **kwargs)
# (57,)
#
# R[1:]  .shape
# Set parameter alpha to: original_alpha * n_samples.
# (57,)
# FutureWarning,
# R[:-1].shape
# (57,)
# n[:-1]  .shape
# (57,)
# R[:-1].shape
# (57,)

# X[1:] - X[:-1] .shape
# (57,)
# R[1:] - R[:-1].shape
# (57,)

# ....

# X.shape
# (371,)
# R.shape
# (371,)
# n.shape
# (371,)
# (X[1:] - X[:-1] + R[1:] - R[:-1]) .shape
# (371,)
# (X[:-1] * (n[:-1] - X[:-1] - R[:-1])).shape
# (370,)

# X[1:] .shape
# (370,)
# X[:-1].shape
# (370,)

# X[1:] .shape
# (370,)
# X[:-1].shape
# (370,)

# R[1:]  .shape
# (370,)
# R[:-1].shape
# (370,)

# n[:-1]  .shape
# (370,)
# R[:-1].shape
# (370,)

# X[1:] - X[:-1] .shape
# (371,)
# R[1:] - R[:-1].shape
# (371,)

S = n - X - R

# 因为他是df 相减 而不是 arr 相减
X_diff = np.array([X[:-1], X[1:]], dtype=np.float64).T
R_diff = np.array([R[:-1], R[1:]], dtype=np.float64).T
# 为什么 4个 2维的东西加起来会变成3维度
gamma = (R[1:] - R[:-1]) / X[:-1]
print(" (X[1:] - X[:-1] + R[1:] - R[:-1]) .shape")
print( (X[1:] - X[:-1] + R[1:] - R[:-1]) .shape )
print("(X[:-1] * (n[:-1] - X[:-1] - R[:-1])).shape")
print((X[:-1] * (n[:-1] - X[:-1] - R[:-1])).shape)
# 去掉了 最前面的和去掉了最后面的 进行一个相减
print("X[1:] .shape")
print(X[1:] .shape)
print(" X[:-1].shape")
print( X[:-1].shape)

# print("X[1:] ")
# print(X[1:] )
# print(" X[:-1]")
# print( X[:-1])

print(" R[1:]  .shape")
print( R[1:]  .shape)
print(" R[:-1].shape")
print( R[:-1].shape)

# print(" R[1:]  ")
# print( R[1:]  )
# print(" R[:-1]")
# print( R[:-1])

print("n[:-1]  .shape")
print(n[:-1]  .shape)
print(" R[:-1].shape")
print( R[:-1].shape)

# print("X[1:] - X[:-1]")
# print(X[1:] - X[:-1])
#
# print("R[1:] - R[:-1]")
# print(R[1:] - R[:-1])

print("X[1:] - X[:-1] .shape")
print((X[1:] - X[:-1]).shape)

print("R[1:] - R[:-1].shape")
print((R[1:] - R[:-1]).shape)

beta = n[:-1] * (X[1:] - X[:-1] + R[1:] - R[:-1]) / (X[:-1] * (n[:-1] - X[:-1] - R[:-1]))
R0 = beta / gamma

########## Parameters for Ridge Regression ##########
##### Orders of the two FIR filters in (12), (13) in the paper. #####
orders_beta = 3
orders_gamma = 3

##### Select a starting day for the data training in the ridge regression. #####
start_beta = 10
start_gamma = 10

########## Print Info ##########
print("\nThe latest transmission rate beta of SIR model:", beta[-1])
print("The latest recovering rate gamma of SIR model:", gamma[-1])
print("The latest basic reproduction number R0:", R0[-1])



print("\nSIR模型的最新传输速率beta:", beta[-1])
print("SIR型号的最新回收率：", gamma[-1])
print("最新的基本复制数 R0：", R0[-1])

########## Ridge Regression ##########
##### Split the data to the training set and testing set #####
x_beta, y_beta = data_spilt(beta, orders_beta, start_beta)
x_gamma, y_gamma = data_spilt(gamma, orders_gamma, start_gamma)

##### Searching good parameters #####
#clf_beta = ridge(x_beta, y_beta)
#clf_gamma = ridge(x_gamma, y_gamma)

##### Training and Testing #####
clf_beta = Ridge(alpha=0.003765, copy_X=True, fit_intercept=False, max_iter=None, normalize=True, random_state=None, solver='auto', tol=1e-08).fit(x_beta, y_beta)
clf_gamma = Ridge(alpha=0.001675, copy_X=True, fit_intercept=False, max_iter=None,normalize=True, random_state=None, solver='auto', tol=1e-08).fit(x_gamma, y_gamma)

beta_hat = clf_beta.predict(x_beta)
gamma_hat = clf_gamma.predict(x_gamma)

##### Plot the training and testing results #####
plt.figure(1)
# plt.plot(y_beta, label=r'$\beta (t)$ 传播率')
# plt.plot(beta_hat, label=r'$\hat{\beta}(t)$')
plt.plot(y_beta, label=r'$\beta (t)$ 传播率')
plt.plot(beta_hat, label=r'$\hat{\beta}(t)$ 模拟传播率')
plt.title(r"传播率 $\beta (t)$是确切的 $\hat{\beta}(t)$是模拟出来的")
plt.legend()

plt.figure(2)
# plt.plot(y_gamma, label=r'$\gamma (t)$')
# plt.plot(gamma_hat, label=r'$\hat{\gamma}(t)$')
plt.plot(y_gamma, label=r'$\gamma (t)$ 确切治愈率')
plt.plot(gamma_hat, label=r'$\hat{\gamma}(t)$ 模拟治愈率')
plt.title(r"治愈率 $\gamma (t)$是确切的 $\hat{\gamma}(t)$是模拟出来的")
plt.legend()

########## Time-dependent SIR model ##########

##### Parameters for the Time-dependent SIR model #####
stop_X = 0 # stopping criteria
stop_day = 100 # maximum iteration days (W in the paper)

day_count = 0
turning_point = 0

S_predict = [S[-1]]
X_predict = [X[-1]]
R_predict = [R[-1]]

predict_beta = np.array(beta[-orders_beta:]).tolist()
predict_gamma = np.array(gamma[-orders_gamma:]).tolist()
while (X_predict[-1] >= stop_X) and (day_count <= stop_day):
    if predict_beta[-1] > predict_gamma[-1]:
        turning_point += 1

    next_beta = clf_beta.predict(np.asarray([predict_beta[-orders_beta:]]))[0]
    next_gamma = clf_gamma.predict(np.asarray([predict_gamma[-orders_gamma:]]))[0]

    if next_beta < 0:
        next_beta = 0
    if next_gamma < 0:
        next_gamma = 0

    predict_beta.append(next_beta)
    predict_gamma.append(next_gamma)

    next_S = ((-predict_beta[-1] * S_predict[-1] *
               X_predict[-1]) / n[-1]) + S_predict[-1]
    next_X = ((predict_beta[-1] * S_predict[-1] * X_predict[-1]) /
              n[-1]) - (predict_gamma[-1] * X_predict[-1]) + X_predict[-1]
    next_R = (predict_gamma[-1] * X_predict[-1]) + R_predict[-1]

    S_predict.append(next_S)
    X_predict.append(next_X)
    R_predict.append(next_R)

    day_count += 1

########## Print Info ##########
print('\nConfirmed cases tomorrow:', np.rint(X_predict[1] + R_predict[1]))
print('Infected persons tomorrow:', np.rint(X_predict[1]))
print('Recovered + Death persons tomorrow:', np.rint(R_predict[1]))

print('\nEnd day:', day_count)
print('Confirmed cases on the end day:', np.rint(X_predict[-2] + R_predict[-2]))

print('\nTuring point:', turning_point)

print('\n明天的确诊数量:', np.rint(X_predict[1] + R_predict[1]))
print('明天的被感染数量:', np.rint(X_predict[1]))
print('明天的治愈加上死亡的数量:', np.rint(R_predict[1]))

print('\n结束的那天:', day_count)
print('结束那天会有多少确诊:', np.rint(X_predict[-2] + R_predict[-2]))

print('\n转折点(第几天):', turning_point)

########## Plot the time evolution of the time-dependent SIR model ##########
plt.figure(3)
# plt.plot(range(len(X) - 1, len(X) - 1 + len(X_predict)), X_predict, '*-', label=r'$\hat{X}(t)$', color='darkorange')
# plt.plot(range(len(X) - 1, len(X) - 1 + len(X_predict)), R_predict, '*-', label=r'$\hat{R}(t)$', color='limegreen')
# plt.plot(range(len(X)), X, 'o--', label=r'$X(t)$', color='chocolate')
# plt.plot(range(len(X)), R, 'o--', label=r'$R(t)$', color='darkgreen')
# plt.xlabel('Day')
# plt.ylabel('Person')


plt.plot(range(len(X) - 1, len(X) - 1 + len(X_predict)), X_predict, '*-', label=r'$\hat{X}(t)$ 预测患病人数', color='darkorange')
plt.plot(range(len(X) - 1, len(X) - 1 + len(X_predict)), R_predict, '*-', label=r'$\hat{R}(t)$ 预测治愈人数', color='limegreen')
plt.plot(range(len(X)), X, 'o--', label=r'$X(t)$ 实际患病人数', color='chocolate')
plt.plot(range(len(X)), R, 'o--', label=r'$R(t)$ 实际治愈人数', color='darkgreen')

plt.xlabel('日期')
plt.ylabel('人数')


plt.title(' 时间相关SIR模型的时间演化。')
# plt.title('Time evolution of the time-dependent SIR model. 时间相关SIR模型的时间演化。')
# 时间相关SIR模型的时间演化。

plt.legend()

plt.show()
