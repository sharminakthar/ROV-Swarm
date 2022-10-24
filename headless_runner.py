import pandas as pd
from flock_settings import FlockSettings, Setting
from headless.parser import Parser
from simulator import Simulator
from util.csv_helpers import *


class HeadlessRunner():
    def __init__(self):
        self.settings = FlockSettings()
        
        self.parser = Parser(self.settings)
        parameters = self.parser.parse_cmd_arguments()

        for (name, value) in parameters.items():
            if(name == "STEPS"):
                self.steps = value    
            elif(name == "SAMPLES"):
                self.samples = value
            elif(type(value) is str):
                #rough support for enum arguments, does mean strings cannot be used without manual checks
                type_of = type(self.settings.get_default(Setting[name]))
                self.settings.set(Setting[name], type_of[value])
            else:
                self.settings.set(Setting[name], value)

    def run(self):
        directory = get_new_output_directory()
        output_metadata(self.settings, directory)

        print("Starting simulation with settings: \n")

        for (name, setting) in Setting.__members__.items():
            value = str(self.settings.get(setting))
            print("\t" + name + " = " + value)

        print("")

        if(self.samples == 1):
            self.execute_single(directory)     
        else:
            self.execute_multiple(directory, self.samples)
        
    def execute_single(self, directory):        
        simulator = Simulator(self.settings)  
        print("starting")
        for step in range(0, self.steps):
            simulator.update(log_data=False) 
            # Commenting out print statement decreases run time         
            #print(f"\rProgress: [ {step+1} / {self.steps} ] simulation steps", end="")

        #output_metrics_log(simulator, directory)
        output_raw_data_log(simulator, directory)
        
    def execute_multiple(self, directory, count):
        metric_logs = []
        
        for sample in range(count):
            self.settings.set(Setting.SEED, sample)
            simulator = Simulator(self.settings) 
            for step in range(0, self.steps):
                simulator.update(log_data=True)          
                print(f"\rSample [ {sample+1} / {count}] Progress: [ {step+1} / {self.steps} ] simulation steps", end="")
            
            metric_logs.append(simulator.get_metrics_log())    
        
        grouped = pd.concat(metric_logs).groupby("Timestep", as_index=False)
        
        output_dataframe_csv(grouped.mean(), directory, "metrics_log_mean.csv")
        output_dataframe_csv(grouped.std(), directory, "metrics_log_std.csv")
        output_dataframe_csv(grouped.min(), directory, "metrics_log_min.csv")
        output_dataframe_csv(grouped.max(), directory, "metrics_log_max.csv")
        
            
            
runner = HeadlessRunner()
runner.run()

