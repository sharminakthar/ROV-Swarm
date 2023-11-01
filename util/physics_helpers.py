from math import sqrt
import numpy as np
from .vector_helpers import angle_radians, normalize, rotate


diameter = 0.16  # m
p = 1020.0  # kg/m^3 approx. mass density of sea water
cd = 0.42  # drag coefficient
a = np.pi * (0.5 * diameter)**2  # m^2 surface area of hemisphere

def calculate_acceleration(force, max_acc, max_dec, crt_vel, timestep):
    acceleration = np.copy(force) # + calculate_drag(crt_vel).reshape((2, 1))

    # the following lines check if the acceleration value of the drone is less than or equal to the max acceleration
    # if it is greater than max_acc, resize the acceleration vector so that it's magnitude is equal to max_acc (while still pointing in same direction)
    acc_magnitude = np.linalg.norm(acceleration)

    current_speed = np.linalg.norm(crt_vel)
    potential_speed = np.linalg.norm(crt_vel + np.copy(acceleration) * timestep)

    new_acceleration = np.copy(acceleration)
    # current and potential speed (new speed after effect of acceleration) used to check if the
    # acceleration vector is a acceleration or a deceleration
    # if acceleration, check that it doesn't exceed max acceleration
    # else, check it doesn't exceed max deceleration
    if (potential_speed > current_speed):
        if acc_magnitude > max_acc:
            new_acceleration = normalize(acceleration) * max_acc
    else:
        if acc_magnitude > max_dec:
            new_acceleration = normalize(acceleration) * max_dec
    return new_acceleration


def calculate_velocity(crt_acc, crt_vel, max_speed, max_turn_rate, timestep):
    # next, we find the new velocity the drone tries to change to by getting the current velocity of the drone and adding the acceleration
    # note that we do not yet set self.velocity -- this will be explained further on
    new_velocity = np.copy(crt_vel)

    new_velocity += np.copy(crt_acc) * timestep  # * 0.5

    # we check that the new velocity value is less than or equal to max_speed
    # if not, we resize it like we resized acceleration earlier

    speed = np.linalg.norm(new_velocity)

    if speed < 0.01:
        return np.zeros((2, 1))

    if speed > max_speed:
        new_velocity = (np.copy(new_velocity)/speed) * max_speed
        
    # get the angle (in radians) between the current velocity and the new velocity
    angle_in_radians, side = angle_radians(
        np.copy(crt_vel), np.copy(new_velocity))

    # if the angle is greater than the max turn rate, we rotate the new velocity vector so that its angle from the current velocity
    # is the same as self.max_turn_rate
    # hence, this final vector value will be at not too great an angle from the current self.velocity, so we can assign this as self.velocity

    if (np.rad2deg(angle_in_radians) > max_turn_rate * timestep):

        if side == 1:
            rotated_velocity = rotate(
                np.copy(new_velocity), (angle_in_radians - np.deg2rad(max_turn_rate * timestep)))
            new_velocity = np.copy(rotated_velocity)
        else:
            rotated_velocity = rotate(
                np.copy(new_velocity), (2 * np.pi - (angle_in_radians-np.deg2rad(max_turn_rate * timestep))))
            new_velocity = np.copy(rotated_velocity)

    return new_velocity


# Adapted from https://physics.stackexchange.com/questions/205513/splitting-up-a-force-into-horizontal-and-vertical-components
def calculate_drag(velocity, mass_density=p, drag_coefficient=cd, surface_area=a):
    """Returns drag force as a numpy array in the opposte direction of travel.
    Based on mass density, drag coefficient and surface area of drone.
    Uses drone velocity which is taken directly from the drone"""
    speed = np.linalg.norm(velocity)
    drag_magnitude = -0.5 * mass_density * drag_coefficient * surface_area
    x_drag_force = drag_magnitude * velocity[0] * speed
    y_drag_force = drag_magnitude * velocity[1] * speed

    drag_force = np.array([x_drag_force, y_drag_force])
    return drag_force / 12  # divide by mass


def get_numerical_velocities(velocity):
    vector_velocity = np.copy(velocity).reshape((2,))
    x_velocity = vector_velocity[0]
    y_velocity = vector_velocity[1]
    speed = sqrt(x_velocity**2 + y_velocity**2)
    return x_velocity, y_velocity, speed
