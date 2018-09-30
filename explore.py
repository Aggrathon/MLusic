"""
    Script for exploting the data
"""

#%% Setup
from matplotlib import pyplot as plt
import numpy as np 
from song import Song
from convert_input import DATA_FILE

#%% Plot delta times
s = Song().read_data(DATA_FILE)
times = s.times[:, 0]
times = times[(times > 0) * (times < np.max(times)*0.75)]
times /= np.min(times)
plt.hist(times, bins=np.arange(0, 200))
plt.title("Delta Times")
plt.show()
plt.hist(times, bins=np.arange(0, 30, 0.1))
plt.title("Delta Times (small)")
plt.show()

#%% Plot GCD
times = s.times[:, 0]
x = np.arange(1, 100, 1)/3000
y = np.asarray([np.sum(np.abs(times - x * np.round(times / x))) for x in x])
lin = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y)[0]
plt.plot(x, y)
plt.plot([x[0], x[-1]], [x[0]*lin[0]+lin[1], x[-1]*lin[0]+lin[1]])
plt.title("GCD abs dev")
plt.show()
y_ = y - x*lin[0] - lin[1]
plt.plot(x, y_)
plt.title("GCD shifted")
plt.show()

#%%
print(6*np.min(s.times[s.times[:, 0] > 0, 0]), ":", 1/0.005)
print(12*np.min(s.times[s.times[:, 0] > 0, 0]), ":", 1/0.01)
print(0.025, ":", 1/0.025)
