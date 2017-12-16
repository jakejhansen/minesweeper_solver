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
    epochs.append(stat[0])
    steps.append(stat[1])
    train_r.append(stat[2])
    loss.append(stat[3])
    winrate.append(stat[4])
	
    
import IPython
IPython.embed()
plt.plot(steps, train_r, steps, loss, steps, winrate)
plt.grid()
plt.legend(["train reward", "loss", "winrate"])
plt.show()
