import numpy as np


class DroneInfo:

    def __init__(self, drone_id, position, heading, speed):
        self.drone_id = drone_id
        self.position = position
        self.heading = heading
        self.speed = speed

    def get_drone_id(self):
        return self.drone_id

    def get_position(self):
        return np.copy(self.position)
    
    def set_position(self, new_position):
        self.position = new_position

    def get_heading(self):
        return self.heading

    def get_speed(self):
        return self.speed