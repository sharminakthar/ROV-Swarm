from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import pickle
import os 

def GetDblAxsGrph():

    err = 'BW'


    with open("Sensor_Errors/RACETRACK/"+err+"/DRT_"+err+".txt", "rb") as myFile:
        data1 = pickle.load(myFile)

    with open("Sensor_Errors/RACETRACK/"+err+"/CN_"+err+".txt", "rb") as myFile:
        data2 = pickle.load(myFile)

    

    for run in range(5):
        fig, ax1 = plt.subplots()
        n = 1000

        x = np.zeros(shape=(len(data1.keys()),n))
        T = np.zeros(shape=(len(data2.keys()),9999))

        y = list(data1.keys())
        y.sort(key=float)

        z = list(data2.keys())
        z.sort(key=float)



        for i in range(len(data2.keys())):
            for j in range(9999):
                T[i][j] = data2[z[i]][str(run)][j]
        
        y2 = np.zeros(shape = len(data2.keys()))

        for i in range(len(data2.keys())):
            y2[i] = np.sum(T[i])


        for i in range(len(data1.keys())):
            for j in range(n):
                x[i][j] = data1[y[i]][str(run)][j+(9999-n)]


        y1 = np.zeros(shape = len(data1.keys()))
        x_axis = np.zeros(shape = len(data2.keys())) 
        #std_error = []       
        for i in range(len(data2.keys())):
            y1[i] = np.mean(x[i])
            x_axis[i] = float(y[i])
         #   std_error.append(np.std(x[i], ddof=1) / np.sqrt(len(x[i])))

    
        ax2 = ax1.twinx()
        ax1.plot(x_axis, y1 , c = 'g')
        ax2.plot(x_axis, y2, c = 'b')

        ax1.set_xlabel('Bandwidth')
        ax1.set_ylabel('Distance from Racetrack', color='g')
        ax2.set_ylabel('Collision Number', color='b')

    plt.show()

    return 0
    



if __name__ == "__main__":

    GetDblAxsGrph()

    print("done")