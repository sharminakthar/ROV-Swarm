import numpy as np
from flock_settings import FlockSettings, Setting
from .drone_info import DroneInfo
from .message import Message
from util.vector_helpers import bearing_to_vector, rotate, normalize
from .objectives import DroneObjective


class DroneController:
    def __init__(self, my_id: int, settings: FlockSettings, initial_position, initial_heading, initial_velocity):
        # set all attributes
        self.approximate_position = initial_position
        self.my_id = my_id

        self.objective = settings.get(Setting.DRONE_OBJECTIVE)
        self.separation_distance = settings.get(Setting.SEPARATION_DISTANCE)
        self.target_pos = np.array(
            [settings.get(Setting.TARGET_X), settings.get(Setting.TARGET_Y)]).reshape(2, 1)
        self.target_radius = settings.get(Setting.TARGET_RADIUS)
        self.target_heading = settings.get(Setting.TARGET_HEADING)

        # attribute keeping track of how communication cycles a drone has not received a message from another drone
        self.lost_counter = 0

        self.last_heading_reading = initial_heading
        self.last_velocity_reading = initial_velocity

        # initialise empty list of data about other drones
        self.flock_info = [None for i in range(
            0, settings.get(Setting.FLOCK_SIZE))]

        self._weight_separation = settings.get(Setting.WEIGHT_SEPARATION)
        self._weight_alignment = settings.get(Setting.WEIGHT_ALIGNMENT)
        self._weight_cohesion = settings.get(Setting.WEIGHT_COHESION)
        self._weight_objective = settings.get(Setting.WEIGHT_OBJECTIVE)

    def receive_message(self, id, range, bearing, payload):
        # now, we use the range with error and the bearing with error to calculate the inferred position of the other drone
        calculated_other_position = self.approximate_position + \
            range * bearing_to_vector(bearing).reshape((2, 1))

       # save the position and heading data
        self.flock_info[id] = DroneInfo(
            id, calculated_other_position, payload.get_heading(), payload.get_speed())

    def update(self, elapsed_time):
        self.update_neighbour_positions_using_speed(elapsed_time)

    def get_next_update_force(self) -> np.ndarray((2, 1)):

        force = self.separation_vector(
            self.separation_distance, self._weight_separation).reshape((2, 1))
        force += self.cohesion_vector(self._weight_cohesion).reshape((2, 1))
        force += self.calculate_alignment(self._weight_alignment).reshape((2, 1))

        if self.objective == DroneObjective.NONE:
            pass
        if self.objective == DroneObjective.TARGET_MOTHERSHIP:
            force += self.mothership_follow_vector(
                self._weight_objective).reshape((2, 1))
        elif self.objective == DroneObjective.FOLLOW_CIRCLE:
            force += self.get_circle_follow_vector(
                self._weight_objective, self.target_pos, self.target_radius, use_flock_center=True).reshape((2, 1))
        elif self.objective == DroneObjective.FIXED_HEADING:
            force += self.get_direction_vector(
                self.target_heading, self._weight_objective).reshape((2, 1))
        elif self.objective == DroneObjective.TARGET_POINT:
            force += self.get_move_to_point_vector(
                self.target_pos, self._weight_objective).reshape((2, 1))

        return normalize(force.reshape((2, 1)))

    def calculate_nearest_neighbour(self):
        ds = []
        for drone_info in self.flock_info:
            if drone_info is None:
                ds.append(float('inf'))
                continue
            ds.append(np.linalg.norm(
                self.approximate_position - drone_info.get_position()))
        min = np.min(ds)
        if (min == float('inf')):
            return None
        else:
            return self.flock_info[ds.index(min)]

    def calculate_alignment(self, strength: float):
        velocity_sum = np.zeros((2, 1))

        for drone_info in self.flock_info:
            if drone_info is None:
                continue
            velocity_sum += bearing_to_vector(drone_info.get_heading())
        return normalize(velocity_sum) * strength

    def separation_vector(self, separation_distance, strength):
        force = np.zeros((2, 1))

        for drone_info in self.flock_info:
            if(drone_info is None):
                continue
            separation = self.approximate_position - drone_info.get_position()
            distance = np.linalg.norm(separation)
            if(distance > 0 and distance < separation_distance * 2):
                direction = separation / distance
                force += direction / distance  # weight by distance

        return force * strength * separation_distance

    def get_flock_center(self):
        positions = []
        positions.append(self.approximate_position)
        for drone_info in self.flock_info:
            if drone_info is None:
                continue
            positions.append(drone_info.get_position())
        return np.average(positions, 0)

    def cohesion_vector(self, strength):
        return normalize(self.get_flock_center() - self.approximate_position) * strength

    def mothership_follow_vector(self, strength):
        # assuming drone[0] is mothership - better pattern possible
        mothership_info = self.flock_info[0]
        if(mothership_info is None):
            return np.zeros((2, 1))
        return self.get_move_to_point_vector(mothership_info.get_position(), strength)

    def update_readings(self, velocity_reading, heading_reading, acceleration, elapsed_time):
        velocity_reading = velocity_reading.reshape(2, 1)

        approx_acceleration = acceleration

        # Updates the approximated position of the drone using Verlet integration with velocity readings
        self.approximate_position += velocity_reading * \
            elapsed_time + 0.5 * approx_acceleration * elapsed_time**2
        self.last_velocity_reading = velocity_reading.reshape(2, 1)
        self.last_heading_reading = heading_reading

    def generate_message(self) -> Message:
        return Message(self.my_id, self.last_heading_reading, np.linalg.norm(self.last_velocity_reading))

    def get_circle_follow_vector(self, strength, center, radius, use_flock_center=False, forecast_radians=0.1):
        pos = self.approximate_position

        if(use_flock_center):
            pos = self.get_flock_center()

        center_vec = normalize(pos - center.reshape(2, 1))
        target_vec = rotate(center_vec, forecast_radians)
        target_pos = center.reshape(2, 1) + target_vec.reshape(2, 1) * radius

        return self.get_move_to_point_vector(target_pos, strength)

    def get_direction_vector(self, heading, strength):
        return strength * bearing_to_vector(heading)

    def get_move_to_point_vector(self, point, strength):
        direction = normalize(point - self.approximate_position)
        return direction * strength

    def get_approximated_position(self):
        return self.approximate_position

    def update_neighbour_positions_using_speed(self, elapsed_time):
        for drone_info in self.flock_info:
            if(drone_info is None):
                continue

            pos = drone_info.get_position()
            heading = drone_info.get_heading()
            distance = drone_info.get_speed() * elapsed_time
            new_pos = pos + distance * bearing_to_vector(heading)

            drone_info.set_position(new_pos)
