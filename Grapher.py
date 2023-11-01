from pathlib import Path
import pandas as pd
from .metrics.BaseMetric import BaseMetric


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
            df = self.get_single_val_data(metric, directory)
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

    def get_all_var_data(self, metric_list: dict[str: BaseMetric], directory: Path, graph_func=None,
                         save_data: bool=True) -> dict[str, dict[str, pd.DataFrame]]:
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
        pass

    def generate_line_charts(self, data_list):
        pass

    def generate_bar_charts(self, data_list):
        pass

    def save_data(self, data_list):
        pass

    def save_graphs(self, figs):
        pass

    



if __name__ == "__main__":
    pass