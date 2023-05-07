# Задача 1. Даны значения величины заработной платы заемщиков банка (zp) и значения их  поведенческого кредитного скоринга (ks): 
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]. 
# Используя математические операции, посчитать коэффициенты линейной регрессии, приняв за X заработную плату
# (то есть, zp - признак), а за y - значения скорингового балла (то есть, ks - целевая
# переменная). Произвести расчет как с использованием intercept, так и без.

import numpy as np
from sklearn.linear_model import LinearRegression

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

b1 = (len(zp)*np.sum(zp*ks) - np.sum(zp) * np.sum(ks))/ (len(zp)*np.sum(zp**2) - np.sum(zp)**2)
print('Коэффициент b1 =', b1)

b0 = np.mean(ks) - b1 * np.mean(zp)
print('Коэффициент b0 =', b0)

y_pred = b0 + (b1 * zp)
print(y_pred)
print()

model = LinearRegression()
zp = zp.reshape(-1, 1)
regres = model.fit(zp, ks)
print(regres.intercept_)
print(regres.coef_)
