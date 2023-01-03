from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import pickle
import os 

def GetDblAxsGrph():

    err = 'PL'


    with open("Sensor_Errors/CIRCLE/"+err+"/DRT_"+err+".txt", "rb") as myFile:
        data1 = pickle.load(myFile)

    with open("Sensor_Errors/CIRCLE/"+err+"/CN_"+err+".txt", "rb") as myFile:
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
        print(z)

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
        for i in range(len(data2.keys())):
            y1[i] = np.mean(x[i])
            x_axis[i] = float(y[i])

    
        ax2 = ax1.twinx()
        ax1.plot(x_axis, y1 , c = 'g')
        ax2.plot(x_axis, y2, c = 'b')

        ax1.set_xlabel('Bandwidth')
        ax1.set_ylabel('Distance from Racetrack', color='g')
        ax2.set_ylabel('Collision Number', color='b')

    plt.show()

    return 0
    

class DAGraph:
  def __init__(self, mission , parameter, M1 ,M2, run):
    self.Mission = mission
    self.parameter = parameter
    self.data1 = self.openfile(M1)
    self.data2 = self.openfile(M2)
    self.run = run
    self.x_axis = self.GettingXA()
    self.y1 = self.getYaxis(M1)
    self.y2 = self.getYaxis(M2)

  def openfile(self, metric):
    with open("Sensor_Errors/"+self.Mission+"/"+self.parameter+"/"+metric+"_"+self.parameter+".txt", "rb") as myFile:
        data = pickle.load(myFile)
    return data


  def GettingXA(self):
    z = list(self.data2.keys())
    z.sort(key=float)

    X_axis = np.zeros(shape = len(self.data2.keys()))
    for i in range(len(self.data2.keys())):
         X_axis[i] = float(z[i])

    return X_axis

  def getYaxis(self, metric):
    n = 1000
    x = np.zeros(shape=(len(self.x_axis),n))
    T = np.zeros(shape=(len(self.x_axis),9999))
    YX = np.zeros(shape = len(self.x_axis))

    y = list(self.data1.keys())
    y.sort(key=float)

    if metric == "DRT" or metric == "DEN" or metric == "CDM":
        for i in range(len(self.x_axis)):
            for j in range(n):
                x[i][j] = self.data1[y[i]][self.run][j+(9999-n)]

        for i in range(len(self.x_axis)):
            YX[i] = np.mean(x[i])
    elif metric == "CN":
        for i in range(len(self.x_axis)):
            for j in range(9999):
                T[i][j] = self.data2[y[i]][self.run][j]

        for i in range(len(self.x_axis)):
            YX[i] = np.sum(T[i])
    return YX

  def MakeGraph(self):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(self.x_axis, self.y1 , c = 'g')
    ax2.plot(self.x_axis, self.y2, c = 'b')

    ax1.set_xlabel('Bandwidth')
    ax1.set_ylabel('Distance from Racetrack', color='g')
    ax2.set_ylabel('Collision Number', color='b')
    plt.show()



if __name__ == "__main__":

    b = DAGraph("RACETRACK","BE","DRT","CN","2")
    print(b.MakeGraph())

    print("done")