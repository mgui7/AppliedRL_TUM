import matplotlib.pyplot as plt
import pandas as pd

table = pd.read_csv('mars_lander_result1.csv' , index_col=[0])
# table consists of mean_reward, total_reward and final_reward

plt.plot(table.iloc[:,[0]])
plt.title('Mean reward')
plt.show()

plt.figure()
plt.plot(table.iloc[:,[1]])
plt.title('Total reward')
plt.show()

plt.figure()
plt.plot(table.iloc[:,[2]])
plt.title('Final reward')
plt.show()
