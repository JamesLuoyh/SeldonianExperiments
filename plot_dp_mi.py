import pandas as pd

df = pd.read_csv('/media/yuhongluo/SeldonianExperimentResults/dp_vs_mi.csv')
df2 = pd.read_csv('/media/yuhongluo/SeldonianExperimentResults/dp_vs_mi_i2.csv')

import matplotlib.pyplot as plt
import numpy as np


# plt.style.use("bmh")

delta_dp = df.delta_dp.values
mi_upper = df.mi_upper.values

delta_dp_2 = df2.delta_dp.values
mi_upper_2 = df2.mi.values

def f(x):
    return np.min(np.stack([np.power(x, 2)/2 + np.power(x, 4)/36 + np.power(x, 6)/288, np.log((2+x)/(2-x)) - 2*x/(2+x)]), axis=0)

psi = 0.668 * f(0.332 * delta_dp) + 0.332 * (0.668 * delta_dp)
plt.scatter(delta_dp, mi_upper, c ="blue", linewidths = 0.1, s=5)
# plt.scatter(delta_dp_2, mi_upper_2, c ="green", linewidths = 0.1, s=5)
plt.plot(delta_dp, psi, c ="red", linewidth=1)#, linewidths = 0.1, s=10)
plt.legend(['$\~{I}_3(Z;S)$', '$\psi(\cdot)$'])

plt.xscale('log')
plt.yscale('log')


plt.xlabel('$\Delta_{DP}$')
# plt.ylabel("Y-axis")
plt.savefig('/media/yuhongluo/SeldonianExperimentResults/dp_vs_mi.pdf', format="pdf")
