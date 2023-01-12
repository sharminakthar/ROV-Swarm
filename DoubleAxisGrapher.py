from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import pickle
import csv
import os 
from MetricsList import units_list, metric_list


class DAG:
    timesteps = 0

    def __init__(self, p: Path, mission , parameter, M1 ,M2, run, error):
        self.Mission = mission
        self.parameter = parameter
        self.metric1 = M1
        self.metric2 = M2
        self.data1 = self.openfile(p, M1)
        self.data2 = self.openfile(p, M2)
        self.x_axis = self.GettingXA(p, M1)
        self.y1 , self.y2 = self.getYaxis(M1,M2)

        if error == None:
            for val in (p /"Metric_Data" /  parameter / M1).iterdir():
                print("VAL: ", val.name)
                self.plotSingleError(p, val.name, M1, M2)
        else:
            for err in error:
                self.plotSingleError(p, err, M1, M2)
        self.MakeGraph(p, M1, M2)

    def openfile(self, p : Path, metric):
        pathname = str(p) + "/Metric_Data/"+self.parameter+"/"+metric
        filenames = os.listdir(pathname)
        data = {}

        for i in filenames:
                d = pd.read_csv(pathname+'/'+i+'/metric_data.csv')
                self.timesteps = np.size(d.iloc[:,0])

                data[i] = d
        
        return data

    def GettingXA(self, p: Path, metric):
        pathname =  str(p) + "/Metric_Data/"+self.parameter+"/"+metric 
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
        T = np.zeros(shape=(len(self.x_axis),self.timesteps))
        YX1 = np.zeros(shape = len(self.x_axis))
        YX2 = np.zeros(shape = len(self.x_axis))

        y = self.x_axis

        
        if metric1 == "col_num":
            for i in range(len(self.x_axis)):
                for j in range(self.timesteps):
                    T[i][j] = self.data2[str(y[i])]['0'][j]

            for i in range(len(self.x_axis)):
                YX1[i] = np.sum(T[i])
        #elif metric1 == "cdm" or metric1 == "speed" or metric1 == "den":
        else:
            for i in range(len(self.x_axis)):
                for j in range(n):
                    x[i][j] = self.data1[str(y[i])]['0'][j+(self.timesteps-n)]

            for i in range(len(self.x_axis)):
                YX1[i] = np.mean(x[i])
        

        if metric2 == "col_num":
            for i in range(len(self.x_axis)):
                for j in range(self.timesteps):
                    T[i][j] = self.data2[str(y[i])]['0'][j]

            for i in range(len(self.x_axis)):
                YX2[i] = np.sum(T[i])

        #elif metric2 == "cdm" or metric2 == "speed" or metric2 == "den":
        else:
            for i in range(len(self.x_axis)):
                for j in range(n):
                    x[i][j] = self.data2[str(y[i])]['0'][j+(self.timesteps-n)]

            for i in range(len(self.x_axis)):
                YX2[i] = np.mean(x[i])

        return YX1,YX2


    def MakeGraph(self,p: Path, M1, M2):

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns1 = ax1.plot(self.x_axis, self.y1 , c = 'g', label = metric_list[M1]["axis_label"])
        lns2 = ax2.plot(self.x_axis, self.y2, c = 'b', label = metric_list[M2]["axis_label"] )

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)

        ax1.set_xlabel(str.title(self.parameter.replace("_", " " )))
        ax1.set_ylabel('Distance from Racetrack / m')
        ax2.set_ylabel('Total number of Collisions')
        plt.title('Effect of varying '+str.title(self.parameter.replace("_", " " ))+' on ' + metric_list[M1]["desc"] + ' and ' + metric_list[M2]["desc"])

        folder = p / "Graphs" / self.parameter / "DoubleAxis" / (M1 + "-" + M2)

        folder.mkdir(parents=True, exist_ok=True)


        fig.savefig(folder / "MultiVal.png", bbox_inches="tight")
        plt.close(fig)

    def plotSingleError(self, p: Path, error, M1, M2):

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        lns1 = ax1.plot(self.data1[error]['0'] , c = 'g', label = metric_list[self.metric1]["axis_label"])
        lns2 = ax2.plot(self.data2[error]['0'], c = 'b', label =metric_list[self.metric2]["axis_label"] )

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left', bbox_to_anchor=(0.25, 1.15))

        ax1.set_xlabel('Timestep/s')
        ax1.set_ylabel(metric_list[self.metric1]["axis_label"]+'/ '+metric_list[self.metric1]["unit"])
        ax2.set_ylabel(metric_list[self.metric2]["axis_label"]+'/ '+metric_list[self.metric2]["unit"])
        plt.title('')
        
        folder = p / "Graphs"/ self.parameter / "DoubleAxis"  / (M1 + "-" +  M2)



        folder.mkdir(parents=True, exist_ok=True)


        newdir = error + ".png"
        fig.savefig(folder / newdir, bbox_inches="tight")





if __name__ == "__main__":

    b = DAG("RACETRACK","HEADING_CALIBRATION_ERROR","sep_mean","distfromRT","0")
    b.plotSingleError('30')

    print("done")