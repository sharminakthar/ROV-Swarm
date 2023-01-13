import itertools
import time
from pathlib import Path

from simulation.objectives import DroneObjective
from flock_settings import FlockSettings, Setting
from headless.parser import Parser
from simulator import Simulator
from util.csv_helpers import *


class MultiRunner():
    def __init__(self, variable_groups, variable_values, steps=10000, samples=5):
        # Dictionary of variables to their range of values
        self.variables = variable_values

        #Steps taken per simulation
        self.steps = steps
        # Samples for each set of parameters
        self.samples = samples

        # Variables groups, where grouped variables will vary together to reduce the number of simulations needed
        self.active_variable_groups = variable_groups

    def run(self, output_file, drone_objective):
        # Maps each variable group to their values
        values = list(map(self.group_to_vals, self.active_variable_groups))
        # Zips grouped values together
        grouped_values = map(lambda x: list(zip(*x)), values)
        # A list of values representing each scenario to test
        scenarios = map(self.flatten, itertools.product(*grouped_values))

        vars = self.flatten(self.active_variable_groups)
        var_names = [v.name for v in vars]
        
        for scenario in scenarios:
            print("Scenario " + str(list(zip(var_names, scenario))))
            settings = FlockSettings()
            settings.set(Setting["DRONE_OBJECTIVE"], drone_objective)

            for var, val in zip(vars, scenario):
                settings.set(var, val)
            
            # Make directory name for a set of simulation parameters
            directory = Path("out")
            directory = directory / output_file / "-".join(var_names)
            for s in scenario:
                directory = directory / str(s)

            directory.mkdir(parents=True, exist_ok=True)
            output_metadata(settings, directory)

            for i in range(self.samples):
                seed = int(time.time())
                settings.set(Setting["SEED"], seed)

                new_dir = directory / str(i)
            
                new_dir.mkdir(parents=True, exist_ok=True)

                self.execute_single(new_dir, settings)  
                
                # Output seed to directory of each file
                with open(new_dir / "seed.txt", "w") as f:
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
    # Stores the range of values to be used when testing
    variables = {
        Setting.FLOCK_SIZE: [i +2 for i in range(10)],
        Setting.BANDWIDTH: [round((i*2)/10 + 0.2,2) for i in range(10)],
        Setting.PACKET_LOSS: [10 * i  for i in range(10)],    
        Setting.SPEED_ERROR: [round(5 * i,2) for i in range(10)],
        Setting.HEADING_ERROR: [round(5 * i,2)  for i in range(10)],
        Setting.RANGE_ERROR: [round(5 * i,2) for i in range(10)],
        Setting.BEARING_ERROR: [round(6 * i,2) for i in range(10)],
        Setting.ACCELERATION_ERROR: [round( i,2) for i in range(10)],
        Setting.SPEED_CALIBRATION_ERROR: [round(i,2) for i in range(10)],
        Setting.HEADING_CALIBRATION_ERROR: [round(i*10,2) + 20 for i in range(5)],
        Setting.RANGE_CALIBRATION_ERROR: [round(i*10,2) + 150 for i in range(4)],
        Setting.BEARING_CALIBRATION_ERROR: [round( i*8,2) for i in range(10)],
        Setting.ACCELERATION_CALIBRATION_ERROR: [round(i,2) + 48 for i in range(1)]
        }

    # To run each variable independently: 
    # for var in ["FLOCK_SIZE", "BANDWIDTH", "PACKET_LOSS", "SPEED_ERROR", "HEADING_ERROR", 
    #             "RANGE_ERROR", "BEARING_ERROR", "ACCELERATION_ERROR", "SPEED_CALIBRATION_ERROR",
    #             "HEADING_CALIBRATION_ERROR", "RANGE_CALIBRATION_ERROR", "BEARING_CALIBRATION_ERROR",
    #             "ACCELERATION_CALIBRATION_ERROR"]:
    #    runner = MultiRunner([[var]])
    #    runner.run("RT Raw Data", DroneObjective.RACETRACK)
    
    # Will run just range calibration error
    # runner = MultiRunner([[Setting.RANGE_CALIBRATION_ERROR]], variables)
    # runner.run("RT Raw Data", DroneObjective.RACETRACK)

    # will run all combinations of flock size and packet loss
    #runner = MultiRunner([["FLOCK_SIZE"],["PACKET_LOSS"]])
    #runner.run("RT Raw Data", DroneObjective.RACETRACK)

    # Will join together flock size and packet loss so that they increase at the same time,
    # and cross with range calibration error
    # runner = MultiRunner([["FLOCK_SIZE", "PACKET_LOSS"], ["RANGE_CALIBRATION_ERROR"]])
    # runner.run("RT Raw Data", DroneObjective.RACETRACK)

