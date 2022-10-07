from math import atan2, sqrt
import numpy as np
from numpy.random import Generator


def angle_radians(vector_1, vector_2):
    """

    Parameters
    ----------
    vector_1 : numpy.ndarray
        The first vector of shape (2,1).
    vector_2 : numpy.ndarray
        The second vector of shape (2,1).

    Returns
    -------
    numpy.float64
        The (smaller) angle in radians between the two vectors.
    int
        This value will equal 1 when the second vector is anti-clockwise
        from the first vector, by the value specified by the angle returned
        as the first returned value. Similarly, this value is -1 when
        the second vector clockwise to the first vector.

    """

    # adapted from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

    # get lengths of both vectors
    length_vector_1 = np.linalg.norm(vector_1)
    length_vector_2 = np.linalg.norm(vector_2)

    # if either vector is of 0 length, return 0 and exit function

    if length_vector_1 == 0 or length_vector_2 == 0:
        return 0, 1

    # get unit vectors of both vectors
    unit_vector_1 = vector_1 / length_vector_1
    unit_vector_2 = vector_2 / length_vector_2

    unit_vector_1_alt = unit_vector_1.reshape(1, 2)[0]
    unit_vector_2_alt = unit_vector_2.reshape(1, 2)[0]

    # calculate dot product
    dot_product = np.dot(unit_vector_1_alt, unit_vector_2_alt)

    # ensure dot_product is between -1 and 1 inclusive
    if dot_product >= -1 and dot_product <= 1:
        angle = np.arccos(dot_product)
    else:
        angle = 0

    # check if angle is nan. This is only the case if vectors are in exact same or exact opposite direction
    if np.isnan(angle):
        a = vector_1[0][0]
        b = vector_1[1][0]
        c = vector_2[0][0]
        d = vector_2[1][0]

        # if product of either x or y values of both vectors are positive, then the vectors are in same direction
        if a*c > 0 or b*d > 0:
            angle = 0  # hence angle is 0
        # if product is negative instead, then vectors are in opposite direction
        elif a*c < 0 or b*d < 0:
            angle = np.pi  # hence angle is π
        # if one or both of the vectors is (0,0) then angle is trivially set to 0
        else:
            angle = 0

    # get matrix of both unit vectors and get the determinant
    matrix = np.vstack((unit_vector_1_alt, unit_vector_2_alt))
    det = np.linalg.det(matrix)

    # if determinant is positive, that means the 2nd vector is <angle> anti-clockwise from the 1st
    # if negative, then its clockwise
    # this information is used when velocity vectors need to be rotated in the update function

    if det > 0:
        side = 1
    else:
        side = -1

    return angle, side


def rotate(vector, radians : float):
    """

    Parameters
    ----------
    vector : numpy.ndarray
        The original vector of shape (2,1).
    radians : numpy.float64
        The angle in radians to rotate the original vector clockwise.

    Returns
    -------
    m : numpy.ndarray
        The rotated vector of shape (2,1).

    """
    # adapted from https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302

    #x = vector[0][0]
    #y = vector[1][0]

    c, s = np.cos(radians), np.sin(radians)

    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, vector)

    return m.reshape((2,1))

# convert bearing (clockwise from north) to direction angle (anti-clockwise from x-axis)
def bearing_degrees_to_direction_radians(degrees : float):
    return np.deg2rad((-degrees + 90) % 360.0)

def direction_radians_to_bearing_degrees(radians : float):
    return (np.rad2deg(-radians) + 90) % 360.0

# anticlockwise angle in radians
def polar_to_cartesian(radius : float, direction_angle_radians : float) -> np.ndarray:
    x = radius * np.cos(direction_angle_radians)
    y = radius * np.sin(direction_angle_radians)
    return np.asarray([x, y]).reshape((2, 1))

def bearing_to_vector(heading_degrees : float):
    return polar_to_cartesian(1, bearing_degrees_to_direction_radians(heading_degrees))

def cartesian_to_polar(vector):
    vector = vector.flatten()
    x = vector[0]
    y = vector[1]
    
    radius = sqrt(x*x + y*y)
    theta = atan2(y,x)
    
    return (radius, theta)

def vector_to_bearing(vector):
    (r, theta) = cartesian_to_polar(vector)
    return direction_radians_to_bearing_degrees(theta)

def normalize(v):
    magnitude = np.linalg.norm(v)
    if(magnitude == 0):
        magnitude = 1
    return (v / magnitude)


def get_random_vector(rng : Generator, max_magnitude : float) -> np.ndarray:
    bearing = rng.random() * 2 * np.pi
    magnitude = rng.random() * max_magnitude
    return polar_to_cartesian(magnitude, bearing)

def get_random_vector_normal(rng : Generator, scale : float) -> np.ndarray:
    x = rng.normal(0, scale)
    y = rng.normal(0, scale)
    return np.asarray([x,y]).reshape((2,1))

def get_distance(a, b) -> float:
    return np.linalg.norm(a - b)
    
