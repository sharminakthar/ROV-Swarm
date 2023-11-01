from turtle import position
import numpy as np
from simulation.flock import Flock
import pandas as pd
from numpy.random import default_rng
from util.vector_helpers import get_random_vector
from flock_settings import Setting, FlockSettings
from copy import copy
from util.metrics_helpers import get_metrics_definitions, get_metrics_formats


class Simulator:

    def __init__(self, settings : FlockSettings, steps: int=0):
        self.steps = steps
        self.step = 0               
        self.flock = self.__create_new_flock(settings)
        self.metric_definitions = get_metrics_definitions(self.flock) 
        self.metric_formats = get_metrics_formats()
        self.metrics_log = pd.DataFrame({})

        self.raw_data_log = {
            "Timestep": np.full(steps * settings.get(Setting.FLOCK_SIZE), 0),
            "Drone ID" : np.full(steps * settings.get(Setting.FLOCK_SIZE), 0),
            "X Position" : np.full(steps * settings.get(Setting.FLOCK_SIZE), 0.0),
            "Y Position" : np.full(steps * settings.get(Setting.FLOCK_SIZE), 0.0),
            "X Velocity" : np.full(steps * settings.get(Setting.FLOCK_SIZE), 0.0),
            "Y Velocity" : np.full(steps * settings.get(Setting.FLOCK_SIZE), 0.0),
            "X Approx Position": np.full(steps * settings.get(Setting.FLOCK_SIZE), 0.0),
            "Y Approx Position": np.full(steps * settings.get(Setting.FLOCK_SIZE), 0.0)
        }
        

    def __create_new_flock(self, settings : FlockSettings) -> Flock:
        drone_num = settings.get(Setting.FLOCK_SIZE)

        rng = default_rng(settings.get(Setting.SEED))

        positions = np.zeros((2, drone_num))

        for i in range(0, drone_num):
            positions[:, i] = get_random_vector(rng, settings.get(Setting.MAX_RANGE) / 2).reshape(2,)

        velocities = np.zeros((2, drone_num))

        for i in range(0, drone_num):
            velocities[:, i] = get_random_vector(rng, settings.get(Setting.MAX_SPEED)).reshape(2,)

        flock = Flock(settings=copy(settings), positions=positions, velocities=velocities)

        return flock       

    def get_metrics_data_frame(self):
        data = {}        
        data["Timestep"] = self.step
        for key, function in self.metric_definitions.items():
            data[key] = [function()]        
        return pd.DataFrame(data=data)

    def update_raw_data(self):

        [x_positions, y_positions] = self.flock.get_positions()
        [x_velocities, y_velocities] = self.flock.get_velocities()
        [x_approx_positions, y_approx_positions] = self.flock.get_approx_positions()

        if self.steps != 0:
            change_range_1 = self.step * self.flock.get_size()
            change_range_2 = (self.step + 1) * self.flock.get_size()
            self.raw_data_log["Timestep"][change_range_1 : change_range_2] = [self.step for _ in range(self.flock.get_size())]
            self.raw_data_log["Drone ID"][change_range_1 : change_range_2] = (list(range(0, self.flock.get_size())))
            self.raw_data_log["X Position"][change_range_1 : change_range_2] = x_positions
            self.raw_data_log["Y Position"][change_range_1 : change_range_2] = y_positions
            self.raw_data_log["X Velocity"][change_range_1 : change_range_2] = x_velocities
            self.raw_data_log["Y Velocity"][change_range_1 : change_range_2] = y_velocities
            self.raw_data_log["X Approx Position"][change_range_1 : change_range_2] = x_approx_positions
            self.raw_data_log["Y Approx Position"][change_range_1 : change_range_2] = y_approx_positions
        else:
            self.raw_data_log["Timestep"] = np.concatenate([self.raw_data_log["Timestep"], np.array([self.step for _ in range(self.flock.get_size())])])
            self.raw_data_log["Drone ID"] = np.concatenate([self.raw_data_log["Drone ID"], np.array(list(range(0, self.flock.get_size())))])
            self.raw_data_log["X Position"] = np.concatenate([self.raw_data_log["X Position"], x_positions])
            self.raw_data_log["Y Position"] = np.concatenate([self.raw_data_log["Y Position"], y_positions])
            self.raw_data_log["X Velocity"] = np.concatenate([self.raw_data_log["X Velocity"], x_velocities])
            self.raw_data_log["Y Velocity"] = np.concatenate([self.raw_data_log["Y Velocity"], y_velocities])
            self.raw_data_log["X Approx Position"] = np.concatenate([self.raw_data_log["X Approx Position"],x_approx_positions])
            self.raw_data_log["Y Approx Position"] = np.concatenate([self.raw_data_log["Y Approx Position"],y_approx_positions])

    def get_metrics_log(self):
        return self.metrics_log

    def get_raw_data_log(self):
        return pd.DataFrame(data=self.raw_data_log)

    def update(self, log_data=False, log_raw_data=True):
        self.flock.update(self.step)
        
        if(log_data):
            self.metrics_log = pd.concat([self.metrics_log , self.get_metrics_data_frame()])
        if(log_raw_data):
            self.update_raw_data()
        
        self.step += 1

    def get_flock(self):
        return self.flock

    def get_step(self):
        return self.step
