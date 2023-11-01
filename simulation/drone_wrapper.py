from math import sqrt
import numpy as np
from flock_settings import FlockSettings, Setting
from simulation.sensors.acceleration_sensor import AccelerationSensor
from simulation.sensors.heading_sensor import HeadingSensor
from simulation.sensors.velocity_sensor import VelocitySensor
from .sensors.bearing_sensor import BearingSensor
from .drone_controller import DroneController
from util.physics_helpers import calculate_acceleration, calculate_velocity
from .message import Message
from numpy.random import default_rng
from util.vector_helpers import normalize, rotate, get_distance, vector_to_bearing
from .sensors.range_sensor import RangeSensor


class DroneWrapper(object):

    def __init__(self, parent_flock, settings: FlockSettings, position, velocity, seed, my_id=0):
        # set all attributes
        self.parent_flock = parent_flock
        self.my_id = my_id

        # attribute keeping track of how communication cycles a drone has not received a message from another drone
        self.lost_counter = 0

        # create random number generator
        self.rng = default_rng(seed)

        # Initialise sensor emulators
        self.range_sensor = RangeSensor(self.rng, settings.get(
            Setting.RANGE_ERROR), settings.get(Setting.RANGE_CALIBRATION_ERROR))
        self.bearing_sensor = BearingSensor(self.rng, settings.get(
            Setting.BEARING_ERROR), settings.get(Setting.BEARING_CALIBRATION_ERROR))
        self.heading_sensor = HeadingSensor(self.rng, settings.get(
            Setting.HEADING_ERROR), settings.get(Setting.HEADING_CALIBRATION_ERROR))
        self.velocity_sensor = VelocitySensor(self.rng, settings.get(
            Setting.SPEED_ERROR), settings.get(Setting.SPEED_CALIBRATION_ERROR))
        self.acceleration_sensor = AccelerationSensor(self.rng, settings.get(
            Setting.ACCELERATION_ERROR), settings.get(Setting.ACCELERATION_CALIBRATION_ERROR))
        self.exact_position = position
        self.velocity = velocity
        self.heading = vector_to_bearing(velocity)

        self.max_acc = settings.get(Setting.MAX_ACCELERATION)
        self.max_dec = settings.get(Setting.MAX_DECELERATION)
        self.max_speed = settings.get(Setting.MAX_SPEED)
        self.max_turn_rate = settings.get(Setting.MAX_RATE_OF_TURN)

        self.drone_controller = DroneController(self.my_id, settings, np.copy(
            self.exact_position), np.copy(self.heading), np.copy(self.velocity))

    def update(self, elapsed_time: float):
        """Updates the position of the drone using Verlet integration
        (https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet)
        This updates the position, velocity and acceleration based on the flocking forces
        and also takes into account drag force acting on the drone
        """
        self.drone_controller.update(elapsed_time)

        force = self.apply_heading_error(
            self.drone_controller.get_next_update_force())

        self.acceleration = calculate_acceleration(
            force, self.max_acc, self.max_dec, self.velocity, elapsed_time)

        self.velocity = calculate_velocity(
            crt_acc=self.acceleration, crt_vel=self.velocity, max_speed=self.max_speed, max_turn_rate=self.max_turn_rate, timestep=elapsed_time).reshape(2, 1)

        self.heading = vector_to_bearing(self.velocity)
        # finally, we update self.exact_position using self.velocity

        self.exact_position = np.copy(self.exact_position) + np.copy(self.velocity) * \
            elapsed_time + 0.5 * np.copy(self.acceleration) * elapsed_time**2

        self.update_controller_information(elapsed_time)

    def apply_heading_error(self, force):
        return rotate(force, np.deg2rad(self.drone_controller.last_heading_reading - self.heading))

    # intended to be overridden by the mothership
    def update_controller_information(self, elapsed_time):
        heading_with_error = self.heading_sensor.get_reading(self.velocity)
        velocity_with_error = self.velocity_sensor.get_reading(
            np.linalg.norm(self.velocity), heading_with_error)
        acceleration_with_error = self.acceleration_sensor.get_reading(
            self.acceleration)

        self.drone_controller.update_readings(
            velocity_with_error, heading_with_error, acceleration_with_error, elapsed_time)

    def get_heading(self):
        return self.heading

    def get_x_y_positions(self):
        position = np.copy(self.exact_position).reshape((2,))
        x_position = position[0]
        y_position = position[1]
        return x_position, y_position

    def generate_message(self) -> Message:
        return self.drone_controller.generate_message()

    def handle_message(self, message: Message):

        other_drone = self.parent_flock.get_drone(message.get_drone_id())

        other_exact_position = other_drone.get_exact_position().reshape((2, 1))

        # get range reading from range sensor
        other_range = self.range_sensor.get_reading(
            self.exact_position, other_exact_position)

        # get the relative bearing using the position in the Message object
        relative_bearing = self.bearing_sensor.get_reading(
            self.exact_position, other_exact_position)

        self.drone_controller.receive_message(
            message.get_drone_id(), other_range, relative_bearing, message)

    def print_data(self):

        ID = 0

        for droneInfo in self.drone_controller.flock_info:

            if droneInfo is None:
                print("Drone has no info on drone ", ID)
            else:
                print("Drone thinks that drone ", ID, " has position:")
                print(droneInfo.get_position())
                print("Along with heading:")
                print(droneInfo.get_heading())

            ID += 1

    def get_numerical_velocities(self):
        vector_velocity = np.copy(self.velocity).reshape((2,))
        x_velocity = vector_velocity[0]
        y_velocity = vector_velocity[1]
        numerical_velocity = sqrt(x_velocity**2 + y_velocity**2)
        return x_velocity, y_velocity, numerical_velocity

    def get_speed(self):
        return np.linalg.norm(self.velocity)

    def get_exact_position(self):
        return self.exact_position

    def is_mothership(self):
        return False

    def get_approximated_position(self):
        return self.drone_controller.get_approximated_position()

    def get_position_error(self):
        return np.linalg.norm(self.exact_position - self.get_approximated_position())

    def get_neighour_error(self):
        total = 0
        count = 0
        for drone_info in self.drone_controller.flock_info:
            if(drone_info is None):
                continue
            count += 1
            relative_pos = drone_info.get_position() - self.get_approximated_position()
            actual_relative = self.parent_flock.get_drone(
                drone_info.get_drone_id()).get_exact_position() - self.exact_position
            total += get_distance(relative_pos, actual_relative)
        if(count == 0):
            return 0
        else:
            return total / count
