from pathlib import Path
import pandas as pd
from metrics.BaseMetric import BaseMetric
from metrics.CentreDistMetric import CentreDistMetric
from metrics.Separation import Separation

from metrics.collisionsnumber import CollisionsNumber
from metrics.Density import Density
from metrics.orientations import OrientationMetric
from metrics.PerceivedPosMetric import PerceivedPosMetric
from metrics.speed import Speed
from metrics.trajectories import TrajectoryMetric

from textwrap import wrap

from matplotlib import pyplot as plt

units_list = {
    "ACCELERATION_CALIBRATION_ERROR": "m/s$^2$",
    "ACCELERATION_ERROR": "m/s$^2$",
    "BANDWIDTH": "B/s",
    "FLOCK_SIZE": "",
    "BEARING_CALIBRATION_ERROR": "$^\circ$",
    "BEARING_ERROR": "$^\circ$",
    "HEADING_CALIBRATION_ERROR": "r",
    "HEADING_ERROR": "r",
    "PACKET_LOSS": "%",
    "RANGE_CALIBRATION_ERROR": "m",
    "RANGE_ERROR": "m",
    "SPEED_CALIBRATION_ERROR": "m/s",
    "SPEED_ERROR": "m/s"
}

class Grapher():
    
    def __init__(self):
        pass

    def get_single_val_data(self, metric: BaseMetric, directory: Path) -> pd.DataFrame:
        """
        Calculates the values of the metric on at a single value.

        Parameters
            metric_list: BaseMetric
                The metric to be run on the data.
            directory: Path
                The directory of the data being read.
        
        Returns
            The dataframe linking the number of the run to the dataframe containing the values of the metric
            at each timestep for each run.
        """
        results = {}
        dirs = filter(lambda x: x.is_dir(), directory.iterdir())
        for i, run in enumerate(dirs):
            run = run / "raw_data_log.csv"
            data = pd.read_csv(run)
            result = metric.calculate(data)
            if i == 0:
                results["Timestep"] = result.iloc[:, 0]
            results[str(i)] = result.iloc[:, 1]
    
        return pd.DataFrame(results)

    def get_single_var_data(self, metric: BaseMetric, directory: Path, reduction="mean") -> dict[str, pd.DataFrame]:
        """
        Takes in a list of metrics and a directory of a single variable and produces the metric data 
        of the simulations in that directory. There will be the data for a single graph for each metric.

        Parameters
            metric_list: BaseMetric
                The metric to be run on the data.
            directory: Path
                The directory of the data being read.
            reduction: str
                The method used to reduce the data of the runs into one list of values. Should either be "mean"
                or "runN" where N is the number of the run to extract.
            
        Returns
            A dictionary linking each variable value to the values of the metric at each timestep.
        """
        ind_var_output = {}
        for ind_var in directory.iterdir():
            df = self.get_single_val_data(metric, ind_var)
            result = None
            if reduction == "mean":
                result = pd.concat([df["Timestep"], df.loc[:, df.columns != "Timestep"].mean(axis=1)], axis=1)
            elif reduction.startswith("run"):
                result = pd.concat([df["Timestep"], df[reduction[3:]]], axis=1)
            elif reduction == "none":
                result = df
            else:
                raise KeyError("Expected one of 'mean' or 'runN' for the data reduction")
            ind_var_output[ind_var.name] = result
        
        return ind_var_output

    def get_all_var_data(self, metric_list: dict, directory: Path, reduction: str="mean", graph_func=None,
                         save_data: bool=True):
        """
        Takes in a list of metrics and a directory of multilpe variables changing and produces the metric data 
        of the simulations in that directory. There will be data for a set of graphs for each variable for each metric.

        Parameters
            metric_list: dict
                A dictionary linking the name of a metric to a metric information.
            directory: Path
                The directory of the data being read.
            reduction: str
                The method used to reduce the data of the runs into one list of values. Should either be "mean"
                or "runN" where N is the number of the run to extract.
            graph_func:
                The function used to produce a graph from the data
            save_data: bool
                A flag that will denote whether or not to save the raw data from the metrics.
        """
        data_folder = Path("out/Metric_Data")
        fig_folder = Path("out/Graphs")
        if save_data:
            data_folder.mkdir(parents=True, exist_ok=True)
        if graph_func:
            fig_folder.mkdir(parents=True, exist_ok=True)
        
        for var in directory.iterdir():
            for metric_name, metric_info in metric_list.items():
                metric = metric_info["instance"]
                name = "{} - {}".format(var.name, metric_name)
                print(name)
                ind_var_output = self.get_single_var_data(metric, var, reduction=reduction)
                if save_data:
                    folder = data_folder / var.name / metric_name    
                    for k, v in ind_var_output.items():
                        f = folder / k
                        f.mkdir(parents=True, exist_ok=True)
                        v.to_csv(path_or_buf=(f / "metric_data.csv"), index=False)

                if graph_func is not None:
                    fig = graph_func(ind_var_output, metric_info, var.name)
                    folder = fig_folder / var.name / metric_name
                    folder.mkdir(parents=True, exist_ok=True)
                    fig.savefig(folder / (metric_name + ".png"), bbox_inches="tight")
                    plt.close(fig)

    def generate_line_chart(self, data, metric_info, var):
        var_name = " ".join([w.capitalize() for w in var.split("_")])
        fig = plt.figure()
        ax = fig.add_subplot()
        labels = sorted(list(data.keys()), key=lambda x: float(x))
        for k in labels:
            d = data[k]
            ax.plot(d["Timestep"].to_numpy(), d.loc[:, d.columns != "Timestep"].to_numpy(), label=k)
        ax.set_title("{} with varying {}".format(metric_info["desc"], var_name), wrap=True)
        ax.set_xlabel("Timesteps (s)")
        ax.set_ylabel("{} ({})".format(metric_info["axis_label"], metric_info["unit"]))
        legend_title = "{} ({})".format(var_name, units_list[var])
        legend_title = "\n".join(wrap(legend_title, 20))
        l = ax.legend(title=legend_title, bbox_to_anchor=(1.04,1), loc="upper left")
        plt.setp(l.get_title(), multialignment='center')
        return fig

    def generate_bar_chart(self, data_list):
        pass



if __name__ == "__main__":
    grapher = Grapher()
    metric_list = {
                    "cdm": {
                        "desc": "Distance between drones and centre of flock",
                        "unit": "m",
                        "axis_label": "Average Distance From the Centre",
                        "instance": CentreDistMetric(),
                        },
                    "sep_min": {
                        "desc": "Minimum separation between drones",
                        "unit": "m",
                        "axis_label": "Minimum Drone Separation",
                        "instance": Separation()
                        },
                   "sep_max": {
                        "desc": "Maximum separation between drones",
                        "unit": "m",
                        "axis_label": "Maximum Drone Separation",
                        "instance": Separation(reduction="max")
                        },
                   "sep_mean": {
                        "desc": "Mean separation between drones",
                        "unit": "m",
                        "axis_label": "Mean Drone Separation",
                        "instance": Separation(reduction="mean")
                        },
                   "col_num": {
                        "desc": "Total number of collisions",
                        "unit": "",
                        "axis_label": "Number of Collisions",
                        "instance": CollisionsNumber()
                        },
                    "density": {
                        "desc": "Density of the swarm",
                        "unit": "m$^2$",
                        "axis_label": "Swarm Density",
                        "instance": Density()
                        },
                    "orient": {
                        "desc": "S.D of drone orientations",
                        "unit": "$^\circ$",
                        "axis_label": "Drone Orientation S.D",
                        "instance": OrientationMetric()
                        },
                    "pos_err": {
                        "desc": "Calculated position error",
                        "unit": "m",
                        "axis_label": "Calculated Position Error",
                        "instance": PerceivedPosMetric()
                        },
                    "speed": {
                        "desc": "Speed of drones",
                        "unit": "m/s",
                        "axis_label": "Speed",
                        "instance": Speed()
                        },
                    # "traj": {
                    #     "desc": "Difference from optimal trajectory",
                    #     "unit": "$^\circ$",
                    #     "axis_label": "Angle From Optimal Trajectory",
                    #     "instance": TrajectoryMetric()
                    #     }
                   }
    p = Path("out/FOLLOW_CIRCLE")
    grapher.get_all_var_data(metric_list, p, graph_func=grapher.generate_line_chart)