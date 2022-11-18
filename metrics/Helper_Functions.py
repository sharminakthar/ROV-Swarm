import pandas as pd
import math
import numpy as np

def getOrientations(data: pd.DataFrame) -> pd.DataFrame:
    xvels = data.iloc[:, 1]
    yvels = data.iloc[:, 2]
    bearing = 0
    bearings = []
    for x in range(len(xvels)):
        if int(yvels[x] )== 0 or int(xvels[x]) == 0:
            angle = 0
        else:
            angle = math.atan(int(yvels[x]) / int(xvels[x]))

        if int(xvels[x]) < 0 and int(yvels[x]) < 0:
            bearing = 180 + (90 - angle)
        elif int(xvels[x]) < 0 and int(yvels[x]) >= 0:  
            bearing = 270 + angle
        elif int(xvels[x]) >= 0 and int(yvels[x]) < 0:
            bearing = (90 + angle)
        elif int(xvels[x]) >= 0 and int(yvels[x]) >= 0:
            bearing = (90 - angle)
        bearings.append(bearing)
    return bearings

#second parameter is the number adjacent values each data point is averaged with

def moving_average(a, n) :
    listed = a.iloc[:,1].tolist()
    ret = np.cumsum(listed, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    startIndex = np.arange(0.0, math.floor((n-1)/2), 1)
    endIndex = np.arange(10000 - (math.ceil((n-1)/2)), 10000, 1)

    print(np.concatenate((startIndex, endIndex)))
    a = a.drop(index=(np.concatenate((startIndex, endIndex))))


    a["0"] =  (ret[n - 1:] / n)
    cols = a.columns[1]
    a = a.drop(columns =cols)
    return  a



    