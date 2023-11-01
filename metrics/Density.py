from pathlib import Path
from .BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

class Density(BaseMetric):

    def __init__(self):
        super().__init__()

    def myfunction(self, data: pd.DataFrame) -> pd.DataFrame:
        density = 0

        x_p = data["X Position"]
        y_p = data["Y Position"]

        x_array = x_p.to_numpy()
        y_array = y_p.to_numpy()

        nw_array = np.zeros((data.shape[0],2))
        nw_array[:,0] = x_array
        nw_array[:,1] = y_array


        hull = ConvexHull(nw_array)
        density = ((hull.volume)/data.shape[0])

        return(density)

    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep","X Position", "Y Position" ]]
        groups = df.groupby("Timestep")

        df= df.groupby('Timestep').apply(self.myfunction).reset_index()
        df = df.rename(columns={0: "Density"})

        return df

if __name__ == "__main__":
    metric = Density()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../FOLLOW_CIRCLE_EXTENDED/FLOCK_SIZE"
    p = Path(path_name)
    data = metric.run_metric(p)
    
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()

    print("done")

