import itertools
import time
from simulation.objectives import DroneObjective
import pandas as pd
from flock_settings import FlockSettings, Setting
from headless.parser import Parser
from simulator import Simulator
from util.csv_helpers import *


class MultiRunner():
    def __init__(self, variable_groups, steps=10000, samples=5):
        # Stores the range of values to be used when testing
        self.variables = {
            "FLOCK_SIZE": [5],
            "BANDWIDTH": [ i*0.3 for i in range(13)],
            "PACKET_LOSS": [10*i for i in range(10)],
            }
        #Steps taken per simulation
        self.steps = steps
        # Samples for each set of parameters
        self.samples = samples

        # Variables groups, where grouped variables will vary together to reduce the number of simulations needed
        self.active_variable_groups = variable_groups

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
            settings.set(Setting["DRONE_OBJECTIVE"], DroneObjective.FOLLOW_CIRCLE)

            for var, val in zip(var_names, scenario):
                settings.set(Setting[var], val)
            
            # Make directory name for a set of simulation parameters
            directory = "out\\FIXED_RACETRACK_MULTIVAR_STEADY\\" + "-".join(var_names) + "\\" + "\\".join(str(s) for s in scenario)
            if not os.path.exists(directory):
                os.makedirs(directory)
            output_metadata(settings, directory)

            for i in range(self.samples):
                seed = int(time.time())
                settings.set(Setting["SEED"], seed)

                new_dir = directory + "\\" + str(i)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)

                self.execute_single(new_dir, settings)  
                
                # Output seed to directory of each file
                with open(new_dir + "\\seed.txt", "w") as f:
                    f.write(str(seed))
        
    def execute_single(self, directory, settings):        
        simulator = Simulator(settings, self.steps)  

        for step in range(0, self.steps):
            simulator.update(log_data=True)          

        output_raw_data_log(simulator, directory)
    
    def flatten(self, groups):
        return [x for group in groups for x in group]
    
    def group_to_vals(self, group):
        """
        Accepts one group of variables, and returns the zipped values for each of those variables
        """
        return [self.variables[var] for var in group]

if __name__ == "__main__":
    # for var in ["FLOCK_SIZE", "BANDWIDTH", "PACKET_LOSS", "SPEED_ERROR", "HEADING_ERROR", 
    #             "RANGE_ERROR", "BEARING_ERROR", "ACCELERATION_ERROR", "SPEED_CALIBRATION_ERROR",
    #             "HEADING_CALIBRATION_ERROR", "RANGE_CALIBRATION_ERROR", "BEARING_CALIBRATION_ERROR",
    #             "ACCELERATION_CALIBRATION_ERROR"]:
    # for var in ["FLOCK_SIZE", "BANDWIDTH", "PACKET_LOSS", "SPEED_ERROR", "HEADING_ERROR", 
    #             "RANGE_ERROR", "BEARING_ERROR", "ACCELERATION_ERROR"]:
    #for var in ["BEARING_CALIBRATION_ERROR", "RANGE_CALIBRATION_ERROR", "SPEED_CALIBRATION_ERROR",
    #            "HEADING_CALIBRATION_ERROR","ACCELERATION_CALIBRATION_ERROR"]:
        #runner = MultiRunner([[var]])
        #runner.run()
    runner = MultiRunner([["FLOCK_SIZE"], ["PACKET_LOSS"], ["BANDWIDTH"]])
    runner.run()