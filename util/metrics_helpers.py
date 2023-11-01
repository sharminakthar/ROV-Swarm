from simulation.flock import Flock
from simulation.objectives import DroneObjective, MothershipObjective


def get_metrics_definitions(flock: Flock):
    metrics = {}

    metrics["Cohesion"] = flock.calculate_cohesion
    metrics["Separate Flock Groups"] = flock.calculate_flock_groups
    metrics["Min Separation"] = flock.get_min_separation
    metrics["Average Separation"] = flock.get_average_separation
    metrics["Max Separation"] = flock.get_max_separation
    metrics["Velocity Variation"] = flock.get_velocity_variation
    metrics["Average Position Error"] = flock.get_average_position_error
    metrics["Average Neighbour Error"] = flock.get_average_neighbour_error
    metrics["Collisions"] = flock.get_num_collisions

    if(flock.drone_objective == DroneObjective.TARGET_MOTHERSHIP):
        metrics["Average Distance to Mother"] = flock.get_average_distance_from_mother

    if(flock.drone_objective == DroneObjective.TARGET_POINT
       or flock.mothership_objective == MothershipObjective.TARGET_POINT):
        metrics["Distance from Target"] = flock.get_distance_from_target
    if(flock.drone_objective == DroneObjective.FIXED_HEADING):
        metrics["Distance from Origin"] = flock.get_average_distance_from_origin
        metrics["Heading Error"] = flock.get_heading_error
    if(flock.drone_objective == DroneObjective.FOLLOW_CIRCLE
       or flock.mothership_objective == MothershipObjective.FOLLOW_CIRCLE):
        metrics["Circle Bearing"] = flock.get_circle_bearing
        metrics["Distance from Circle"] = flock.get_distance_from_circle

    return metrics


def get_metrics_formats():
    def percent(value):
        return f"{value*100:.2f}%"

    def int(value):
        return str(value)

    def distance(value):
        return f"{value:.2f}m"

    def degrees(value):
        return f"{value:.2f}°"

    formats = {}

    formats["Cohesion"] = percent
    formats["Separate Flock Groups"] = int
    formats["Min Separation"] = percent
    formats["Average Separation"] = percent
    formats["Max Separation"] = percent
    formats["Velocity Variation"] = percent
    formats["Average Position Error"] = distance
    formats["Average Neighbour Error"] = distance
    formats["Collisions"] = int

    # objective specific metrics
    formats["Average Distance to Mother"] = distance
    formats["Distance from Origin"] = distance
    formats["Heading Error"] = degrees
    formats["Circle Bearing"] = degrees
    formats["Distance from Circle"] = distance
    formats["Distance from Target"] = distance

    return formats
