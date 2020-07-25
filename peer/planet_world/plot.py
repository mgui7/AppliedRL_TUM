import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set(color_codes=True)
path = 'planet_world_result_time_150.csv'
table = pd.read_csv(path)

surviving_time = np.array(table.iloc[:,[5]])
new_time = []

for i in range(0,int(len(surviving_time)/2),2):
    new_time.append(float(surviving_time[i][0]))

sns.lineplot(range(len(new_time)),new_time)
plt.xlabel('Iterations')
plt.ylabel('Surviving time')

plt.xlim(0,len(new_time))
plt.ylim(0,325)
plt.show()


surviving_time = np.array(table.iloc[:,[1]])
new_time = []

for i in range(0,int(len(surviving_time)/2),2):
    new_time.append(float(surviving_time[i][0]))

sns.lineplot(range(len(new_time)),new_time)
plt.xlabel('Iterations')
plt.ylabel('Mean Reward')

plt.xlim(0,len(new_time))
# plt.ylim(0,325)
plt.show()
