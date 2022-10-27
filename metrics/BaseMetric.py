import pandas as pd
from abc import ABC, abstractmethod
import os
import numpy as np

class BaseMetric(ABC):
    """
    Abstract class defining key methods for every metric
    """

    def __init__(self):
        pass

    def read_csv(self, csv_file: str) -> pd.DataFrame:
        pass

    def run_metric(self, folder):
        """
        
        Returns
            A list of dataframes, where each dataframe holds the results of repeated runs from one variable value. The
            dataframes contain one column for timestep, and n columns for eachr run. 
        """
        ind_var_output = []
        print(folder)
        print(os.listdir(folder))
        for ind_var in os.listdir(folder):
            ind_var = folder + "\\" + ind_var
            results = {}
            dirs = os.listdir(ind_var)
            dirs = list(filter(lambda x: x != "metadata.csv", dirs))
            for i, run in enumerate(dirs):
                run = ind_var + "\\" + run + "\\" + "raw_data_log.csv"
                data = pd.read_csv(run)
                result = self.calculate(data)
                if i == 0:
                    results["Timestep"] = result.iloc[:, 0]
                results[str(i)] = result.iloc[:, 1]
            
            ind_var_output.append(pd.DataFrame(results))
        
        return ind_var_output

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the metric on the raw data of a single simulation

        Parameters:
            data: pd.Dataframe
                The raw data to run the metric on
        
        Returns:
            A pandas dataframe with one column for the timestep and another column for the metric.
        """
        pass


    def max(self, data: pd.DataFrame) -> pd.DataFrame:
        return np.max(data.iloc[:, 1])

    def min(self, data: pd.DataFrame) -> pd.DataFrame:
        return np.min(data.iloc[:, 1])


    def std(self, data: pd.DataFrame) -> pd.DataFrame:
        return np.std(data.iloc[:, 1])


    def mean(self, data: pd.DataFrame) -> pd.DataFrame:
        return np.mean(data.iloc[:, 1])

    def median(self, data: pd.DataFrame) -> pd.DataFrame:
        return np.median(data.iloc[:, 1])


    def mode(self, data: pd.DataFrame) -> pd.DataFrame:
        vals, counts = np.unique(data.iloc[:, 1], return_counts=True)
        return np.argwhere(counts == np.max(counts))
       


