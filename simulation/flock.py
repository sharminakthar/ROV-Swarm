import math
from typing import List
import numpy as np
from numpy.lib.function_base import average
from numpy.random import SeedSequence, default_rng

from flock_settings import FlockSettings, Setting
from simulation.message import Message
from util.vector_helpers import get_distance, vector_to_bearing
from .drone_wrapper import DroneWrapper
from .mothership_wrapper import MothershipWrapper
from .objectives import MothershipObjective, DroneObjective


# this implementation adapts implementations found on both
# https://alan-turing-institute.github.io/rsd-engineeringcourse/ch01data/084Boids.html
# and
# https://medium.com/better-programming/boids-simulating-birds-flock-behavior-in-python-9fff99375118
class Flock:
    """Flock implementation keeping track of all drones"""
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    drones: List[DroneWrapper] = None
    mother_id = 0

    def __init__(self, settings: FlockSettings, positions, velocities):

        # set attributes using arguments passed to constructor
        self.flock_size = settings.get(Setting.FLOCK_SIZE)
        self.message = None

        self.packet_loss = settings.get(Setting.PACKET_LOSS)
        self.separation_distance = settings.get(Setting.SEPARATION_DISTANCE)

        self.bandwidth = settings.get(Setting.BANDWIDTH)
        self.message_size = settings.get(Setting.MESSAGE_SIZE)
        self.com_range = settings.get(Setting.MAX_RANGE)

        self.collision_distance = settings.get(
            Setting.SEPARATION_DISTANCE) / 10

        self.seed = settings.get(Setting.SEED)
        self.rng = default_rng(self.seed)
        self.master_seed = SeedSequence(self.seed)

        child_seeds = self.master_seed.spawn(self.flock_size)

        self.__collision_count = 0
        self.__last_collisions = []

        self.__messages_sent_last_step = []

        self.mothership_objective = settings.get(Setting.MOTHERSHIP_OBJECTIVE)
        self.drone_objective = settings.get(Setting.DRONE_OBJECTIVE)
        self.target_pos = np.array(
            [settings.get(Setting.TARGET_X), settings.get(Setting.TARGET_Y)]).reshape(2, 1)
        self.target_radius = settings.get(Setting.TARGET_RADIUS)
        self.target_heading = settings.get(Setting.TARGET_HEADING)

        is_mothership_mission = (self.mothership_objective != MothershipObjective.NONE
                                 or self.drone_objective == DroneObjective.TARGET_MOTHERSHIP)

        self.drones = []

        for i in range(self.flock_size):
            if(is_mothership_mission and i == 0):
                mothership = MothershipWrapper(self, settings, position=positions[:, 0].reshape(
                    (2, 1)), velocity=velocities[:, 0].reshape((2, 1)), seed=child_seeds[0])
                self.drones.append(mothership)
            else:
                drone = DroneWrapper(self, settings, positions[:, i].reshape(
                    (2, 1)), velocities[:, i].reshape((2, 1)), seed=child_seeds[i], my_id=i)
                self.drones.append(drone)

    def get_size(self):
        return self.flock_size

    def get_positions(self):
        """This function returns the positions of all the drones in the flock as a 2xcount array."""
        return np.asarray([d.get_exact_position() for d in self.drones]).T.reshape((2, self.flock_size))
    
    def get_approx_positions(self):
        """This function returns the approximate positions of all the drones in the flock as a 2xcount array."""
        return np.asarray([d.get_approximated_position() for d in self.drones]).T.reshape((2, self.flock_size))

    def get_speeds(self):
        """This function returns the sppeds of all the drones in the flock as a list"""
        return [np.linalg.norm(d.velocity) for d in self.drones]

    def get_velocities(self):
        """This function returns the velocities of all the drones in the flock as a 2xcount array"""
        return np.asarray([d.velocity for d in self.drones]).T.reshape((2, self.flock_size))

    def get_headings(self):
        """This function returns the headings of all the drones in the flock as an list of size flock_size"""
        return [d.heading for d in self.drones]

    def update(self, step):
        messages = self.get_messages_for_step(step)

        for message in messages:
            sender_id = message.get_drone_id()
            for drone in self.drones:
                receiver_id = drone.my_id
                if(sender_id != receiver_id and self.drones_in_range(sender_id, receiver_id)):
                    if(self.rng.random() * 100 >= self.packet_loss):
                        drone.handle_message(message)

        self.__messages_sent_last_step = messages

        for drone in self.drones:
            drone.update(1)

        self.check_for_collisions()

    def get_messages_for_step(self, step) -> list[Message]:
        messages = []

        total_messages_sent = self.get_total_messages_sent_by_step(step)
        messages_already_sent = self.get_total_messages_sent_by_step(step - 1)

        for message_index in range(messages_already_sent, total_messages_sent):
            sender_index = message_index % self.flock_size
            message = self.get_drone(sender_index).generate_message()
            messages.append(message)

        return messages

    def get_total_messages_sent_by_step(self, step):
        return max(0, int(math.floor(step * self.bandwidth / self.message_size)))

    def get_messages_sent_last_step(self) -> list[Message]:
        return self.__messages_sent_last_step

    def print_drones_data(self):
        for drone in self.drones:
            drone.print_data()

    def calculate_cohesion(self):
        """Calculate cohesion metric for the swarm,
        returns the desired separation distance over the average distance between the drones and the swarm center
        measured in km^-1"""

        if len(self.drones) < 2:
            return 1

        mean = [sum(x)/len(x)
                for x in zip(*[d.get_exact_position() for d in self.drones])]

        total = 0

        for d in self.drones:
            total += get_distance(mean, d.get_exact_position())

        average_dist = total / len(self.drones)

        return np.sqrt(len(self.drones)) * self.separation_distance / average_dist / 4

    def get_min_separation(self):
        if len(self.drones) < 2:
            return 0

        nearest_dist = np.Infinity

        for drone_a in self.drones:
            for drone_b in self.drones:
                if drone_a != drone_b:
                    dist = get_distance(
                        drone_a.get_exact_position(), drone_b.get_exact_position())
                    if(dist < nearest_dist):
                        nearest_dist = dist

        return nearest_dist / self.separation_distance

    def get_average_separation(self):
        if len(self.drones) < 2:
            return 0
        sum = 0
        count = 0

        for drone_a in self.drones:
            nearest_dist = np.Infinity
            for drone_b in self.drones:
                if drone_a != drone_b:
                    dist = get_distance(
                        drone_a.get_exact_position(), drone_b.get_exact_position())
                    if dist < nearest_dist:
                        nearest_dist = dist
            sum += nearest_dist
            count += 1

        return (sum / count) / self.separation_distance

    def get_max_separation(self):
        if len(self.drones) < 2:
            return 0

        max = 0

        for drone_a in self.drones:
            nearest_dist = np.Infinity
            for drone_b in self.drones:
                dist = get_distance(
                    drone_a.get_exact_position(), drone_b.get_exact_position())
                if dist != 0 and dist < nearest_dist:
                    nearest_dist = dist
            if(nearest_dist > max):
                max = nearest_dist

        return max / self.separation_distance

    def get_velocity_variation(self):
        if len(self.drones) < 2:
            return 0

        mean_velocity = [sum(x)/len(x)
                         for x in zip(*[d.velocity for d in self.drones])]

        sum_of_differences_from_mean = 0

        speeds = [np.linalg.norm(d.velocity) for d in self.drones]
        mean_speed = sum(speeds)/len(speeds)

        for d in self.drones:
            sum_of_differences_from_mean += get_distance(
                mean_velocity, d.velocity)

        return (sum_of_differences_from_mean / len(self.drones)) / mean_speed

    def get_drone(self, drone_id):
        return self.drones[drone_id]

    def drones_in_range(self, drone_a_id, drone_b_id):
        drone_a = self.drones[drone_a_id]
        drone_b = self.drones[drone_b_id]

        dist = get_distance(drone_a.get_exact_position(),
                            drone_b.get_exact_position())

        return dist <= self.com_range

    def get_average_position_error(self):
        return average([d.get_position_error() for d in self.drones])

    def get_average_neighbour_error(self):
        return average([d.get_neighour_error() for d in self.drones])

    def calculate_flock_groups(self):
        tagged_drones = [False] * self.flock_size
        to_expand = []
        groups = 0

        while(not all(tagged_drones)):
            if(len(to_expand) > 0):
                selection = to_expand.pop()
            else:
                selection = next(index for index, tagged in enumerate(
                    tagged_drones) if not tagged)
                tagged_drones[selection] = True
                groups += 1
            for i in range(0, self.flock_size):
                if(not tagged_drones[i] and self.drones_in_range(selection, i)):
                    tagged_drones[i] = True
                    to_expand.append(i)

        return groups

    def check_for_collisions(self):
        collisions = []

        for a_index, drone_a in enumerate(self.drones):
            for b_index, drone_b in enumerate(self.drones):
                # prevent self and duplicates
                if(a_index < b_index):
                    dist = get_distance(
                        drone_a.get_exact_position(), drone_b.get_exact_position())
                    if(dist < self.collision_distance):
                        collision = (a_index, b_index)
                        collisions.append(collision)
                        if(not collision in self.__last_collisions):
                            self.__collision_count += 1
        self.__last_collisions = collisions

    def get_num_collisions(self):
        return self.__collision_count

    def get_average_distance_from_mother(self):
        mother = self.drones[0]
        return average([get_distance(mother.get_exact_position(), d.get_exact_position())
                        for d in self.drones[1:]])

    def get_average_distance_from_origin(self):
        return np.linalg.norm(self.get_flock_center())

    def get_heading_error(self):
        pos = self.get_flock_center()
        heading = vector_to_bearing(pos)
        error = abs(heading - self.target_heading)
        if(error > 180):
            error = 360 - error
        return error

    def get_flock_center(self):
        return np.average([d.get_exact_position() for d in self.drones], axis=0)

    def get_distance_from_target(self):
        return get_distance(self.get_flock_center(), self.target_pos)

    def get_distance_from_circle(self):
        return abs(self.target_radius - self.get_distance_from_target())

    def get_circle_bearing(self):
        vec = self.get_flock_center() - self.target_pos
        return vector_to_bearing(vec)
