import pandas as pd

df = pd.read_csv('/media/yuhongluo/SeldonianExperimentResults/lmifr_backup.csv')
# df = pd.read_csv('/media/yuhongluo/SeldonianExperimentResults/ablation_backup.csv')

import matplotlib.pyplot as plt
import numpy as np


# plt.style.use("bmh")

delta_dp = df.delta_dp.values
mi_upper = df.mi_upper.values

df = df[(df['delta_dp'] <= 0.08)]# & (df['delta_dp'] >= 0.074)] #&  (df['delta_dp'] >= 0.07)
df = df.sort_values(by=['auc'], ascending=False)
# df = df.sort_values(by=['mi_upper'])
print(df.iloc[0:10])
