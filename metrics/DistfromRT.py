from pathlib import Path
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import math
import csv

class distfromRT(BaseMetric):

    def __init__(self):
        super().__init__()

    def myfunction(self, data: pd.DataFrame) -> pd.DataFrame:
        x = data.shape[0]
        data = data.sum(axis = 0)
        swarmcentre = data/x
        if swarmcentre["X Position"]<2000:
            distance = abs(1000 - math.hypot(2000 - swarmcentre["X Position"], 3000 - swarmcentre["Y Position"]))
        elif swarmcentre["X Position"]>3500:
            distance = abs(100 - math.hypot(3500 - swarmcentre["X Position"], 3000 - swarmcentre["Y Position"]))
        elif swarmcentre["Y Position"]<3000:
            distance = abs(2000 - swarmcentre["Y Position"])
        elif swarmcentre["Y Position"]>3000:
            distance = abs(4000 - swarmcentre["Y Position"])
        return(distance)

    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep","X Position", "Y Position" ]]
        df= df.groupby('Timestep').apply(self.myfunction).reset_index()
        df = df.rename(columns={0: "Distance"})
        return df

if __name__ == "__main__":
    metric = distfromRT()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS"
    p = Path(path_name)
    data = metric.run_metric(p)
    print('running')

    with open("DRT_PL.txt", "wb") as myFile:
        pickle.dump(data, myFile)
    
    
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()

    print("done")