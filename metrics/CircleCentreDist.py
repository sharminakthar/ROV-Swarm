from pathlib import Path
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

class circlecentedist(BaseMetric):

    def __init__(self):
        super().__init__()

    def myfunction(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sum(axis = 0)
        data = data/data.shape[0]
        print(data.shape[0])
        distance = math.dist([data["X Position"],data["Y Position"]], [2500,2500])
        return(distance)

    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep","X Position", "Y Position" ]]

        df= df.groupby('Timestep').apply(self.myfunction).reset_index()
        df = df.rename(columns={0: "Distance"})

        return df

if __name__ == "__main__":
    metric = circleCentreDist()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/FLOCK_SIZE"
    p = Path(path_name)
    data = metric.run_metric(p)
    
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()

    print("done")