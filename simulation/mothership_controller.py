from flock_settings import FlockSettings, Setting
from util.vector_helpers import normalize
from .drone_controller import DroneController
import numpy as np
from .objectives import MothershipObjective


class MothershipController(DroneController):

    def __init__(self, my_id: int, settings: FlockSettings, initial_position, initial_heading, initial_velocity):
        super().__init__(my_id, settings, initial_position, initial_heading, initial_velocity)

        self.objective = settings.get(Setting.MOTHERSHIP_OBJECTIVE)

    def update_information_with_exact(self, position, heading, velocity):
        self.approximate_position = position
        self.last_heading_reading = heading
        self.last_velocity_reading = velocity

    def get_next_update_force(self) -> np.ndarray((2,1)):
        
        force = np.zeros((2,1))
        
        if self.objective == MothershipObjective.FOLLOW_CIRCLE:               
            force = self.get_circle_follow_vector(4, self.target_pos, self.target_radius).reshape(2,1)
        elif self.objective == MothershipObjective.TARGET_POINT:   
            force = self.get_move_to_point_vector(self.target_pos, 4).reshape(2,1)

        return normalize(force)    
