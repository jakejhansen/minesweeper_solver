import numpy as np
import pandas as pd
import IPython
import matplotlib.pyplot as plt
import pickle 
epochs = []
steps = []
train_r = []
loss = []
winrate = []

stats = pickle.load(open("stats.p", "rb"))
for stat in stats:
    steps.append(stat[1])
    train_r.append(stat[2]/1.5)
    loss.append(stat[3])
    winrate.append(stat[4])
	
   
plt.rcParams.update({'font.size': 33}) 
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(steps, train_r, steps, loss)
ax1.grid()
ax1.legend(["train reward", "loss"])

ax2.plot(steps, winrate, linewidth = 4.0)
ax2.legend(["win rate"])
plt.xlabel('Steps / Actions')
ax2.grid()


import IPython
IPython.embed()
plt.show()
