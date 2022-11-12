from pathlib import Path
import pandas as pd
from metrics.BaseMetric import BaseMetric
from metrics.CentreDistMetric import CentreDistMetric
from metrics.Separation import Separation

from matplotlib import pyplot as plt


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
                or "runN" where N is the number of the run to extract
            
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

    def get_all_var_data(self, metric_list: dict[str: BaseMetric], directory: Path, reduction="mean", graph_func=None,
                         save_data: bool=True, *graph_args) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Takes in a list of metrics and a directory of multilpe variables changing and produces the metric data 
        of the simulations in that directory. There will be data for a set of graphs for each variable for each metric.

        Parameters
            metric_list: dict[str, BaseMetric]
                A dictionary linking the name of a metric to a metric instance.
            directory: Path
                The directory of the data being read.
        
        Returns
            A dictionary linking each variable name to a metric dictionary, which links each metric name to a data dictionary,
            which links each variable value to the values of the metric at each timestep (a bit disgusting i know).
        """
        data_folder = Path("out/Metric_Data")
        fig_folder = Path("out/Graphs")
        if save_data:
            data_folder.mkdir(parents=True, exist_ok=True)
        if graph_func:
            fig_folder.mkdir(parents=True, exist_ok=True)
        
        for var in directory.iterdir():
            for metric_name, metric in metric_list.items():
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
                    fig = graph_func(ind_var_output)
                    folder = fig_folder / var.name / metric_name
                    folder.mkdir(parents=True, exist_ok=True)
                    fig.savefig(folder / (metric_name + ".png"))

    def generate_bar_chart(self, data):
        fig = plt.figure()
        for k,d in data.items():
            plt.plot(d["Timestep"].to_numpy(), d.loc[:, d.columns != "Timestep"].to_numpy(), label=k, figure=fig)
        return fig

    def generate_line_chart(self, data_list):
        pass

    def save_data(self, data_list):
        pass

    def save_graphs(self, figs):
        pass



if __name__ == "__main__":
    grapher = Grapher()
    metric_list = {"cdm": CentreDistMetric(), "sep_min": Separation(), "sep_max": Separation(reduction="max")}
    p = Path("out/FOLLOW_CIRCLE")
    grapher.get_all_var_data(metric_list, p, graph_func=grapher.generate_bar_chart)