from flock_settings import FlockSettings
from simulator import Simulator


class Controller():

    def __init__(self):
        self.settings = FlockSettings()

        self.sim_speed = 1

        self.dynamic_view_enabled = True
        self.show_comms_ranges = False
        self.show_message_propagtion = False
        self.show_position_approximation = False
        self.show_neigbour_approximations = False

        self.__preview_min_x = -1000
        self.__preview_min_y = -1000
        self.__preview_max_x = 5000
        self.__preview_max_y = 5000

        self.simulator = self.__create_simulator()

    def __create_simulator(self):
        self.settings.cache_active_settings()
        return Simulator(self.settings)

    def reset(self):
        self.simulator = self.__create_simulator()    

    def update(self):
        for i in range(0, self.sim_speed):
            self.simulator.update(log_data=i==0)

    def try_set_preview_min_x(self, value):
        self.__preview_min_x = min(value, self.__preview_max_x - 1)
        return self.__preview_min_x

    def try_set_preview_max_x(self, value):
        self.__preview_max_x = max(value, self.__preview_min_x + 1)
        return self.__preview_max_x

    def try_set_preview_min_y(self, value):
        self.__preview_min_y = min(value, self.__preview_max_y - 1)
        return self.__preview_min_y

    def try_set_preview_max_y(self, value):
        self.__preview_max_y = max(value, self.__preview_min_y + 1)
        return self.__preview_max_y

    def get_preview_min_x(self):
        return self.__preview_min_x

    def get_preview_max_x(self):
        return self.__preview_max_x

    def get_preview_min_y(self):
        return self.__preview_min_y

    def get_preview_max_y(self):
        return self.__preview_max_y