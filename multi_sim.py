import itertools
import pandas as pd
from flock_settings import FlockSettings, Setting
from headless.parser import Parser
from simulator import Simulator
from util.csv_helpers import *


class MultiRunner():
    def __init__(self):
        # Stores the range of values to be used when testing
        self.variables = {
            "FLOCK_SIZE": [3 + i for i in range(5)],
            "BANDWIDTH": [4 * i for i in range(1,4)],
            "PACKET_LOSS": [4*i for i in range(1,4)],
            "SPEED_ERROR": [round(0.6 * i,2) for i in range(10)],
            "HEADING_ERROR": [round(0.02 * i,2) for i in range(10)],
            "RANGE_ERROR": [round(0.6 * i,2) for i in range(10)],
            "BEARING_ERROR": [round(1.6 * i,2) for i in range(10)],
            "ACCELERATION_ERROR": [round(0.1 * i,2) for i in range(10)],
            "SPEED_CALIBRATION_ERROR": [round(0.02 * i,2) for i in range(10)],
            "HEADING_CALIBRATION_ERROR": [round(0.1 * i,2) for i in range(10)],
            "RANGE_CALIBRATION_ERROR": [round(0.2 * i,2) for i in range(10)],
            "BEARING_CALIBRATION_ERROR": [round(0.1 * i,2) for i in range(10)],
            "ACCELERATION_CALIBRATION_ERROR": [round(0.01 * i,2) for i in range(10)]
            }
        #Steps taken per simulation
        self.steps = 100
        # Samples for each set of parameters
        self.samples = 3

        # Variables groups, where grouped variables will vary together to reduce the number of simulations needed
        self.active_variable_groups = [["FLOCK_SIZE"], ["SPEED_ERROR", "ACCELERATION_ERROR"], ["HEADING_ERROR", "BEARING_ERROR"]]#,
                                       #["RANGE_ERROR"], ["BANDWIDTH", "PACKET_LOSS"]]

    def run(self):
        # Maps each variable group to their values
        values = list(map(self.group_to_vals, self.active_variable_groups))
        # Zips grouped values together
        grouped_values = map(lambda x: list(zip(*x)), values)
        # A list of values representing each scenario to test
        scenarios = map(self.flatten, itertools.product(*grouped_values))

        var_names = self.flatten(self.active_variable_groups)
        for scenario in scenarios:
            print("Scenario " + str(list(zip(var_names, scenario))))
            settings = FlockSettings()
            for var, val in zip(var_names, scenario):
                settings.set(Setting[var], val)
            
            directory = "out\\" + "-".join(var_names) + "\\" + "\\".join(str(s) for s in scenario)
            if not os.path.exists(directory):
                os.makedirs(directory)
            output_metadata(settings, directory)

            for i in range(self.samples):
                new_dir = directory + "\\" + str(i)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                self.execute_single(new_dir, settings)     
        
    def execute_single(self, directory, settings):        
        simulator = Simulator(settings)  

        for step in range(0, self.steps):
            simulator.update(log_data=True)          
            print(f"\rProgress: [ {step+1} / {self.steps} ] simulation steps", end="")

        output_raw_data_log(simulator, directory)
    
    def flatten(self, groups):
        return [x for group in groups for x in group]
    
    def group_to_vals(self, group):
        """
        Accepts one group of variables, and returns the zipped values for each of those variables
        """
        return [self.variables[var] for var in group]

if __name__ == "__main__":
    runner = MultiRunner()
    runner.run()
