import numpy as np
import pandas as pd
import IPython
import matplotlib.pyplot as plt
import pickle 
data = pd.read_csv('training_stats.csv', header=None)
epoch = []
train_r = []
loss = []
for x in range(len(data[0])):
    epoch.append(float(data[0][x][0:6]))
    train_r.append(float(data[0][x][23:]))
    loss.append(float(data[2][x][6:]))
    
epoch_start = epoch[-1]
stats = pickle.load(open("stats.p", "rb"))
for stat in stats:
    epoch.append(stat[0] + epoch_start)
    train_r.append(stat[1])
    loss.append(stat[3])
    
plt.plot(epoch, train_r, epoch, loss)
plt.grid()
plt.legend(["train reward", "loss"])
plt.show()
stats[0]