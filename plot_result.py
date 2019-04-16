import json
import matplotlib.pyplot as plt
import numpy as np

def errorfill(x, y, yerr, color='r', alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def TotalReward(rec):
    return np.array(rec).sum()

def SmoothReward(rec, smooth=20):
    rlist = []
    vlist = []
    for i in range(len(rec)-smooth):
        temp = np.asarray(rec[i:i+smooth])
        rlist.append(np.mean(temp))
        vlist.append(np.std(temp))
    return rlist, vlist

def PlotResult(fname, color, smooth=20):
    f = open(fname, "r")
    rec = json.load(f)

    rlist = []
    for i in range(len(rec)):
        rlist.append(TotalReward(rec[i]))
    slist, vlist = SmoothReward(rlist, smooth)
    #plt.plot(rlist, 'b')
    #plt.plot(slist, 'r')
    errorfill(range(len(slist)), np.array(slist), np.array(vlist), color=color)

#PlotResult("rec_map_sensor_baseline.json", 'r')
#PlotResult("rec_map_baseline.json", 'b')
PlotResult("rec.json", 'r')
plt.show()