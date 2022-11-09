from pathlib import Path
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

class OrientationMetric(BaseMetric):

    def __init__(self):
        super().__init__()



    def getOrientations(self, data: pd.DataFrame) -> pd.DataFrame:
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
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Velocity", "Y Velocity"]]
        bearings = self.getOrientations(df)
        df = df[["Timestep"]]
        new_df = df.assign(Orientations = bearings)
        groups = new_df.groupby("Timestep").std()
        df = df.merge(groups, on="Timestep")
        return df

   

if __name__ == "__main__":
    metric = OrientationMetric()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/FLOCK_SIZE"
    p = Path(path_name)

    data = metric.run_metric(p)
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()