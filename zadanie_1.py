# Задача 1. Даны значения величины заработной платы заемщиков банка (zp) и значения их  поведенческого кредитного скоринга (ks): 
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]. 
# Используя математические операции, посчитать коэффициенты линейной регрессии, приняв за X заработную плату
# (то есть, zp - признак), а за y - значения скорингового балла (то есть, ks - целевая
# переменная). Произвести расчет как с использованием intercept, так и без.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

b1 = (len(zp)*np.sum(zp*ks) - np.sum(zp) * np.sum(ks))/ (len(zp)*np.sum(zp**2) - np.sum(zp)**2)
print('Коэффициент b1 =', round(b1, 2))

b0 = np.mean(ks) - b1 * np.mean(zp)
print('Коэффициент b0 =', round(b0, 2))
print()

y_pred = b0 + (b1 * zp)
df = pd.DataFrame({'реальный скоринг': ks, 'предсказанный': y_pred})
print(df)
print()

model = LinearRegression()
zp = zp.reshape(-1, 1)
regres = model.fit(zp, ks)
print('Коэффициент b1 из модели =', round(*regres.coef_, 2))
print('Коэффициент b0 через интерсепт =', round(regres.intercept_, 2))
print()

mse = ((ks - y_pred)**2).sum()/ len(ks)
print('Функция потерь mse =', round(mse, 2))
print('Коэффициент детерминации R2 =', round(regres.score(zp, ks), 2))

