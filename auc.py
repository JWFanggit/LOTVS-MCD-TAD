import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

df = pd.read_pickle("")
y = [item[1] for item in df[0]]
# t = pd.read_excel("F:/test/labels/100.xlsx", index_col=0)[4:]
t = pd.read_csv("")
# print(t)
arry = np.array(y).reshape(-1, 1)
nor_s = 1-MinMaxScaler().fit_transform(arry)
# arry = pd.DataFrame(nor_s)

def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    s_temp = []
    s_temp.append(s[0])
    # print(s_temp)
    for i in range(1, len(s), 1):
        s_temp.append(alpha * s[i-1] + (1 - alpha) * s_temp[i-1])
    return s_temp

df = pd.DataFrame()

for i in range(1, 10):
    alpha = i / 10
    s = exponential_smoothing(alpha, nor_s)  # 你需要定义exponential_smoothing函数
    ss = [item[0] for item in s]
    ss_array = np.array(ss)
    # normalized_ss = 1 - (ss_array - np.min(ss_array)) / (np.max(ss_array) - np.min(ss_array))
    df[str(i)] = ss_array

# 添加'label'列
df['label'] = t['0']  # 假设t['0']是你的标签数据

# 保存到Excel文件
df.to_excel('', index=False)




