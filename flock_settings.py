from enum import Enum
import time
from copy import deepcopy
from simulation.objectives import DroneObjective, MothershipObjective


class FlockSettings:

    def __init__(self):
        self.dictionary = {}
        self.dictionary_defaults = {}
        self.dictionary_cached = {}

        self.set_defaults()
        self.check_settings()
        self.cache_active_settings()

    def get(self, setting):
        if(setting in self.dictionary):            
            return self.dictionary[setting]
        else:
            return self.dictionary_defaults[setting]

    def get_default(self, setting):
        return self.dictionary_defaults[setting]

    def get_cached(self, setting):
        if(setting in self.dictionary_cached):            
            return self.dictionary_cached[setting]
        else:
            return self.dictionary_defaults[setting]

    def set(self, setting, value):
        self.dictionary[setting] = value

    def set_as_default(self, setting, value):
        self.dictionary_defaults[setting] = value

    def cache_active_settings(self):
        self.dictionary_cached = deepcopy(self.dictionary)

    def reset(self, setting):
        self.dictionary[setting] = self.dictionary_defaults[setting]

    def set_defaults(self):
        self.set_as_default(Setting.SEED, int(time.time()))
        self.set_as_default(Setting.FLOCK_SIZE, 5)

        self.set_as_default(Setting.MAX_SPEED, 8)
        self.set_as_default(Setting.MAX_ACCELERATION, 4)
        self.set_as_default(Setting.MAX_DECELERATION, 4)
        self.set_as_default(Setting.MAX_RATE_OF_TURN, 25)
        self.set_as_default(Setting.MOTHERSHIP_MAX_SPEED, 7.5)

        self.set_as_default(Setting.MAX_RANGE, 1500)
        self.set_as_default(Setting.BANDWIDTH, 4)
        self.set_as_default(Setting.MESSAGE_SIZE, 8)
        self.set_as_default(Setting.PACKET_LOSS, 0)

        self.set_as_default(Setting.SPEED_ERROR, 3)
        self.set_as_default(Setting.HEADING_ERROR, 0.1)
        self.set_as_default(Setting.RANGE_ERROR, 3)
        self.set_as_default(Setting.BEARING_ERROR, 8)
        self.set_as_default(Setting.ACCELERATION_ERROR, .5)

        self.set_as_default(Setting.SPEED_CALIBRATION_ERROR, .1)
        self.set_as_default(Setting.HEADING_CALIBRATION_ERROR, .5)
        self.set_as_default(Setting.RANGE_CALIBRATION_ERROR, 1)
        self.set_as_default(Setting.BEARING_CALIBRATION_ERROR, .5)
        self.set_as_default(Setting.ACCELERATION_CALIBRATION_ERROR, .05)

        self.set_as_default(Setting.SEPARATION_DISTANCE, 75)

        self.set_as_default(Setting.DRONE_OBJECTIVE, DroneObjective.NONE)
        self.set_as_default(Setting.MOTHERSHIP_OBJECTIVE, MothershipObjective.NONE)

        self.set_as_default(Setting.TARGET_X, 2500)
        self.set_as_default(Setting.TARGET_Y, 2500)
        self.set_as_default(Setting.TARGET_RADIUS, 1000)
        self.set_as_default(Setting.TARGET_HEADING, 325)

        self.set_as_default(Setting.WEIGHT_SEPARATION, 1.0)
        self.set_as_default(Setting.WEIGHT_ALIGNMENT, 1.0)
        self.set_as_default(Setting.WEIGHT_COHESION, 2.0)
        self.set_as_default(Setting.WEIGHT_OBJECTIVE, 1.0)


    # check that there is a default for every setting
    def check_settings(self):
        for setting in Setting:
            if(not setting in self.dictionary_defaults):
                print("WARNING: Missing default value for setting: " + str(setting))
    

class Setting(Enum):
    SEED = 1,

    FLOCK_SIZE = 1001
    
    MAX_SPEED = 2001
    MAX_ACCELERATION = 2002
    MAX_DECELERATION = 2003
    MAX_RATE_OF_TURN = 2004
    
    MOTHERSHIP_MAX_SPEED = 2101

    MAX_RANGE = 3001
    BANDWIDTH = 3002
    MESSAGE_SIZE = 3003
    PACKET_LOSS = 3004

    SPEED_ERROR = 4001
    HEADING_ERROR = 4002
    RANGE_ERROR = 4003
    BEARING_ERROR = 4004
    ACCELERATION_ERROR = 4005

    SPEED_CALIBRATION_ERROR = 5001
    HEADING_CALIBRATION_ERROR = 5002
    RANGE_CALIBRATION_ERROR = 5003
    BEARING_CALIBRATION_ERROR = 5004
    ACCELERATION_CALIBRATION_ERROR = 5005

    SEPARATION_DISTANCE = 6001

    DRONE_OBJECTIVE = 7001
    MOTHERSHIP_OBJECTIVE = 7002

    # mission specific settings
    TARGET_X = 7101
    TARGET_Y = 7102
    TARGET_RADIUS = 7103
    TARGET_HEADING = 7104

    WEIGHT_SEPARATION = 8001
    WEIGHT_ALIGNMENT = 8002
    WEIGHT_COHESION = 8003
    WEIGHT_OBJECTIVE = 8004

