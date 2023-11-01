from flock_settings import FlockSettings, Setting
from .drone_wrapper import DroneWrapper
from .mothership_controller import MothershipController


class MothershipWrapper(DroneWrapper):
    def __init__(self, parent_flock, settings : FlockSettings, position, velocity, seed):

        DroneWrapper.__init__(self, parent_flock, settings, position, velocity, seed, my_id=0)

        self.max_speed = settings.get(Setting.MOTHERSHIP_MAX_SPEED)

        self.drone_controller = MothershipController(
            0, settings, position, self.heading, velocity)

    def update_controller_information(self, elapsed_time):
        self.drone_controller.update_information_with_exact(
            self.exact_position, self.heading, self.velocity)

    def is_mothership(self):
        return True
