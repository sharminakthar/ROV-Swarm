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

    def __init__(self, settings : FlockSettings):
        self.step = 0               
        self.flock = self.__create_new_flock(settings)
        self.metric_definitions = get_metrics_definitions(self.flock) 
        self.metric_formats = get_metrics_formats()
        self.metrics_log = pd.DataFrame({})
        self.raw_data_log = pd.DataFrame({})
        

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

    def get_raw_data_frame(self):
        [x_positions, y_positions] = self.flock.get_positions()
        [x_velocities, y_velocities] = self.flock.get_velocities()

        data = {
            "Timestep": [self.step] * self.flock.get_size(),
            "Drone ID" : list(range(0, self.flock.get_size())),
            "X Position" : x_positions,
            "Y Position" : y_positions,
            "X Velocity" : x_velocities,
            "Y Velocity" : y_velocities,
        }

        return pd.DataFrame(data=data)

    def get_metrics_log(self):
        return self.metrics_log

    def get_raw_data_log(self):
        return self.raw_data_log

    def update(self, log_data=False):
        self.flock.update(self.step)
        
        if(log_data):
            # self.metrics_log = pd.concat([self.metrics_log , self.get_metrics_data_frame()])
            self.raw_data_log = pd.concat([self.raw_data_log, self.get_raw_data_frame()])
        
        self.step += 1

    def get_flock(self):
        return self.flock

    def get_step(self):
        return self.step
