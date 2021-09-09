#!/usr/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 根據自己的log_loss.txt中的行數修改lines, 修改訓練時的迭代起始次數(start_ite)和结束次数(end_ite)。
lines = 45090
start_ite = 170 # log_loss.txt裡面的最小迭代次數
end_ite = 45000  # log_loss.txt裡面的最大迭代次數
step = 500  # 跳行數，決定畫圖的稠密程度
igore = 200  # 當開始的loss較大時，你需要忽略前igore次迭代，注意這裡是迭代次數

y_ticks = [i*0.5 for i in range(1,6)]  # 縱坐標的值，可以自己設置。
data_path = 'train_log_loss.txt'  # log_loss的路徑。

####-----------------只需要改上面的，下面的可以不改動
names = ['loss', 'avg', 'rate', 'seconds', 'images']
result = pd.read_csv(data_path, skiprows=[x for x in range(lines) if
                                          (x < lines * 1.0 / ((end_ite - start_ite) * 1.0) * igore or x % step != 9)],
                     error_bad_lines = False, names=names)
result.head()
for name in names:
    result[name] = result[name].str.split(' ').str.get(1)

result.head()
result.tail()

for name in names:
    result[name] = pd.to_numeric(result[name])
result.dtypes
print(result['loss'].values)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

###-----------設置橫坐標的值。
x_num = len(result['loss'].values)
tmp = (end_ite - start_ite - igore) / (x_num * 1.0)
x = []
for i in range(x_num):
    x.append(i * tmp + start_ite + igore)
# print(x)
print('total = %d\n' % x_num)
print('start = %d, end = %d\n' % (x[0], x[-1]))
###----------


ax.plot(x, result['loss'].values, label='avg_loss')
# ax.plot(result['loss'].values, label='loss')
plt.yticks(y_ticks)  # 如果不想自己設置縱坐標，可以注釋掉。
plt.grid()
ax.legend(loc='best')
ax.set_title('Loss curves')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
fig.savefig('loss')