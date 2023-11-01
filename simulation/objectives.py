from enum import Enum


class DroneObjective(Enum):
    NONE = 1
    TARGET_MOTHERSHIP = 2
    FOLLOW_CIRCLE = 3
    FIXED_HEADING = 4,
    TARGET_POINT = 5,

class MothershipObjective(Enum):
    NONE = 0
    FOLLOW_CIRCLE = 1,
    TARGET_POINT = 2,   
    
