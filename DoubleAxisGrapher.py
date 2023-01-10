from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import pickle
import csv
import os 

class DAG:
    metric_list = {
        'cdm':{
            "desc": "Distance between drones and centre of flock",
            "unit": "m",
            "axis_label": "Average Distance From the Centre",
        },
        "sep_min": {
            "desc": "Minimum separation between drones",
            "unit": "m",
            "axis_label": "Minimum Drone Separation",
        },
        "sep_max": {
                "desc": "Maximum separation between drones",
                "unit": "m",
                "axis_label": "Maximum Drone Separation",
        },
        "sep_mean": {
                "desc": "Mean separation between drones",
                "unit": "m",
                "axis_label": "Mean Drone Separation",
        },
        "col_num": {
                "desc": "Total number of collisions",
                "unit": "",
                "axis_label": "Number of Collisions",
                },
        "density": {
                "desc": "Density of the swarm",
                "unit": "m$^2$",
                "axis_label": "Swarm Density",
                },
        "orient": {
            "desc": "S.D of drone orientations",
            "unit": "$^\circ$",
            "axis_label": "Drone Orientation S.D",
            },
        "pos_err": {
            "desc": "Calculated position error",
            "unit": "m",
            "axis_label": "Calculated Position Error",
            },
        "speed": {
            "desc": "Speed of drones",
            "unit": "m/s",
            "axis_label": "Speed",
            },
        "traj": {
            "desc": "Difference from optimal trajectory",
            "unit": "$^\circ$",
            "axis_label": "Angle From Optimal Trajectory",
            },
        "distfromRT":{
            "desc": "Distance from the racetrack",
            "unit": "m",
            "axis_label": "Distance",
        }
        
    }
    def __init__(self, mission , parameter, M1 ,M2, run):
        self.Mission = mission
        self.parameter = parameter
        self.metric1 = M1
        self.metric2 = M2
        self.data1 = self.openfile(M1)
        self.data2 = self.openfile(M2)
        self.x_axis = self.GettingXA(M1)
        self.y1 , self.y2 = self.getYaxis(M1,M2)

    def openfile(self, metric):
        pathname = "Metric_Data/"+self.parameter+"/"+metric
        filenames = os.listdir(pathname)
        data = {}

        for i in filenames:
            d = pd.read_csv("Metric_Data/"+self.parameter+"/"+metric+'/'+i+'/metric_data.csv')
            data[i] = d

        return data

    def GettingXA(self, metric):
        pathname = pathname = "Metric_Data/"+self.parameter+"/"+metric
        filenames = os.listdir(pathname)
        for x in range(len(filenames)):
            if filenames[x].isdigit():
                filenames[x] = int(filenames[x])
            else:
                filenames[x] = float(filenames[x])
        
        filenames = sorted(filenames)

        return filenames

    def getYaxis(self, metric1,metric2):
        n = 1000
        x = np.zeros(shape=(len(self.x_axis),n))
        T = np.zeros(shape=(len(self.x_axis),9999))
        YX1 = np.zeros(shape = len(self.x_axis))
        YX2 = np.zeros(shape = len(self.x_axis))

        y = self.x_axis

        if metric1 == "col_num":
            for i in range(len(self.x_axis)):
                for j in range(9999):
                    T[i][j] = self.data2[str(y[i])]['0'][j]

            for i in range(len(self.x_axis)):
                YX1[i] = np.sum(T[i])
        else:
            for i in range(len(self.x_axis)):
                for j in range(n):
                    x[i][j] = self.data1[str(y[i])]['0'][j+(9999-n)]

            for i in range(len(self.x_axis)):
                YX1[i] = np.mean(x[i])

        if metric2 == "col_num":
            for i in range(len(self.x_axis)):
                for j in range(9999):
                    T[i][j] = self.data2[str(y[i])]['0'][j]
            for i in range(len(self.x_axis)):
                YX2[i] = np.sum(T[i])
        else:
            for i in range(len(self.x_axis)):
                for j in range(n):
                    x[i][j] = self.data2[str(y[i])]['0'][j+(9999-n)]

            for i in range(len(self.x_axis)):
                YX2[i] = np.mean(x[i])


        return YX1,YX2

    def singlegraph(self, number):

        if number == 1:
            for i in self.x_axis:
                self.data1[str(i)]['0'].plot()
        elif number == 2:
            for i in self.x_axis:
                self.data2[str(i)]['0'].plot()
        else:
            print("invalid number: please input 1 or 2")

        plt.show()


    def MakeGraph(self):
        print(self.x_axis)
        print(self.y1)
        print(self.y2)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns1 = ax1.plot(self.x_axis, self.y1 , c = 'g', label = 'Distance from Racetrack')
        lns2 = ax2.plot(self.x_axis, self.y2, c = 'b', label ='Collision Number' )

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)

        ax1.set_xlabel(str.title(self.parameter.replace("_", " " )))
        ax1.set_ylabel('Distance from Racetrack / m')
        ax2.set_ylabel('Total number of Collisions')
        plt.title('Effect of varying '+str.title(self.parameter.replace("_", " " ))+' on the distance \nfrom the racetrack and the total number of collisions')
        plt.show()

    def plotSingleError(self, error):

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        lns1 = ax1.plot(self.data1[error]['0'] , c = 'g', label = self.metric_list[self.metric1]["axis_label"])
        lns2 = ax2.plot(self.data2[error]['0'], c = 'b', label =self.metric_list[self.metric2]["axis_label"] )

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left', bbox_to_anchor=(0.25, 1.15))

        ax1.set_xlabel('Timestep/s')
        ax1.set_ylabel(self.metric_list[self.metric1]["axis_label"]+'/ '+self.metric_list[self.metric1]["unit"])
        ax2.set_ylabel(self.metric_list[self.metric2]["axis_label"]+'/ '+self.metric_list[self.metric2]["unit"])
        plt.title('')
        plt.show()

        plt.show()



if __name__ == "__main__":

    b = DAG("RACETRACK","HEADING_CALIBRATION_ERROR","sep_mean","col_num","0")
    b.MakeGraph()

    print("done")