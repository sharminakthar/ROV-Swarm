from pathlib import Path
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


class Grapher(BaseMetric):

    def __init__(self):
        super().__init__()
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Position", "Y Position"]]
        groups = df.groupby("Timestep")
        centres = groups.mean().reset_index()
        centres = centres.rename(columns={"X Position": "X Centre", "Y Position": "Y Centre"})
        df = df.merge(centres, on="Timestep")
        distances = np.sqrt(((df["X Position"] - df["X Centre"]).pow(2) + (df["Y Position"] - df["Y Centre"]).pow(2)))
        df["Distances"] = distances
        df = df[["Timestep", "Distances"]].groupby("Timestep").mean().reset_index()
        return df

if __name__ == "__main__":
    metric = Grapher()

    path_name = "/Users/yusufkhalidahmedfazlee/Desktop/GDP/GIT/swarm-simulator/out/FLOCK_SIZE"
    p = Path(path_name)

    data = metric.run_metric(p)

    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    
    handles, labels = plt.gca().get_legend_handles_labels()

    order = [5,4,6,2,1,3,0]

    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.title('Mean distance from the centre of the flock as time \n goes by with varying flock size')
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Distance', fontsize=14)

    plt.show()

    # metric_Bandwidth = Grapher()

    # path_name_Bandwidth = "/Users/yusufkhalidahmedfazlee/Desktop/GDP/GIT/swarm-simulator/out/BANDWIDTH"
    # p_Bandwidth = Path(path_name_Bandwidth)

    # data_Bandwidth = metric_Bandwidth.run_metric(p_Bandwidth)

    # for k,d in data_Bandwidth.items():
    #     plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [8,6,3,5,4,7,9,2,1,0]

    # plt.title('Mean distance from the centre of the flock as time \n goes by with varying Bandwidth')
    # plt.xlabel('Timestep', fontsize=14)
    # plt.ylabel('Distance', fontsize=14)
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # plt.show()

    # metric_PACKET_LOSS = Grapher()

    # path_name_PACKET_LOSS = "/Users/yusufkhalidahmedfazlee/Desktop/GDP/GIT/swarm-simulator/out/PACKET_LOSS"
    # p_PACKET_LOSS = Path(path_name_PACKET_LOSS)

    # data_PACKET_LOSS = metric_PACKET_LOSS.run_metric(p_PACKET_LOSS)


    # for k,d in data_PACKET_LOSS.items():
    #     plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [2,8,4,6,1,9,5,0,7,3]

    # plt.title('Mean distance from the centre of the flock as time \n goes by with varying Packet Loss')
    # plt.xlabel('Timestep', fontsize=14)
    # plt.ylabel('Distance', fontsize=14)
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # plt.show()
