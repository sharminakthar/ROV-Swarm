from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import pickle
import os 

def GetDblAxsGrph():


    with open("Sensor_Errors/HE/DRT_HE.txt", "rb") as myFile:
        data1 = pickle.load(myFile)

    with open("Sensor_Errors/HE/CN_HE.txt", "rb") as myFile:
        data2 = pickle.load(myFile)

    for run in range(5):

        x = np.zeros(shape=(10,1000))
        T = np.zeros(shape=(10,9999))

        y = list(data1.keys())
        y.sort(key=float)

        z = list(data2.keys())
        z.sort(key=float)

        print(type(data2['0'][str(run)][9999]))

        for i in range(10):
            for j in range(9999):
                T[i][j] = data2[z[i]][str(run)][j]
        
        y2 = np.zeros(shape = (10))
        for i in range(10):
            y2[i] = np.sum(T[i])


        for i in range(10):
            for j in range(1000):
                x[i][j] = data1[y[i]][str(run)][j+8999]


        y1 = np.zeros(shape = (10))
        x_axis = np.zeros(shape = 10)        
        for i in range(10):
            y1[i] = np.mean(x[i])
            x_axis[i] = int(y[i])

        print(x_axis)
        print(y2)

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.scatter(x_axis, y1 , c = 'g')
        ax2.plot(x_axis, y2, 'b')

    ax1.set_xlabel('Heading Error')
    ax1.set_ylabel('Distance from Racetrack', color='g')
    ax2.set_ylabel('Collision Number', color='b')

    plt.show()

    return 0
    



if __name__ == "__main__":

    GetDblAxsGrph()

    print("done")

