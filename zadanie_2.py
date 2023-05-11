# Задача 2 Посчитать коэффициент линейной регрессии при заработной плате (zp), 
# используя градиентный спуск (без intercept).

import numpy as np

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

def mse_(b1, y = ks, x = zp, n = len(zp)):
    return np.sum((b1 * x - y)**2)/ n

alpha = 1e-6
b1 = 6

for i in range (500):
    b1 -= alpha * (2/len(zp)) * np.sum((b1 * zp - ks) * zp)
    if i % 25 == 0:
        print('b1 = {}, mse = {}'.format(b1, mse_(b1)))

