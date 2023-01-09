from metrics.BaseMetric import BaseMetric
from metrics.CentreDistMetric import CentreDistMetric
from metrics.Separation import Separation
from metrics.collisionsnumber import CollisionsNumber
from metrics.Density import Density
from metrics.orientations import OrientationMetric
from metrics.PerceivedPosMetric import PerceivedPosMetric
from metrics.speed import Speed
from metrics.Fixed_Heading_traj import FHTrajectoryMetric
#from metrics.trajectories import TrajectoryMetric

from metrics.trajectories import TPTrajectoryMetric
#from metrics.trajectories import FHTrajectoryMetric
from metrics.DistfromRT import distfromRT

units_list = {
    "ACCELERATION_CALIBRATION_ERROR": "m/s$^2$",
    "ACCELERATION_ERROR": "m/s$^2$",
    "BANDWIDTH": "B/s",
    "FLOCK_SIZE": "",
    "BEARING_CALIBRATION_ERROR": "$^\circ$",
    "BEARING_ERROR": "$^\circ$",
    "HEADING_CALIBRATION_ERROR": "$^\circ$",
    "HEADING_ERROR": "$^\circ$",
    "PACKET_LOSS": "%",
    "RANGE_CALIBRATION_ERROR": "m",
    "RANGE_ERROR": "m",
    "SPEED_CALIBRATION_ERROR": "m/s",
    "SPEED_ERROR": "m/s"
}

metric_list = {                  
                    "sep_min": {
                        "desc": "Minimum separation between drones",
                        "unit": "m",
                        "axis_label": "Minimum Drone Separation",
                        "instance": Separation()
                        },
                        "cdm": {
                        "desc": "Distance between drones and centre of flock",
                        "unit": "m",
                        "axis_label": "Average Distance From the Centre",
                        "instance": CentreDistMetric(),
                        },

                #    "sep_max": {
                #         "desc": "Maximum separation between drones",
                #         "unit": "m",
                #         "axis_label": "Maximum Drone Separation",
                #         "instance": Separation(reduction="max")
                #         },
                #    "sep_mean": {
                #         "desc": "Mean separation between drones",
                #         "unit": "m",
                #         "axis_label": "Mean Drone Separation",
                #         "instance": Separation(reduction="mean")
                #        },
                #    "col_num": {
                #         "desc": "Total number of collisions",
                #          "unit": "",
                #         "axis_label": "Number of Collisions",
                #         "instance": CollisionsNumber()
                #        },
                    #"density": {
                    #     "desc": "Density of the swarm",
                    #     "unit": "m$^2$",
                    #     "axis_label": "Swarm Density",
                    #     "instance": Density()
                    #    },
                    # "orient": {
                    #     "desc": "S.D of drone orientations",
                    #     "unit": "$^\circ$",
                    #     "axis_label": "Drone Orientation S.D",
                    #     "instance": OrientationMetric()
                    #     },
                    # "pos_err": {
                    #     "desc": "Calculated position error",
                    #     "unit": "m",
                    #     "axis_label": "Calculated Position Error",
                    #     "instance": PerceivedPosMetric()
                    #     },
                    # "speed": {
                    #    "desc": "Speed of drones",
                    #     "unit": "m/s",
                    #     "axis_label": "Speed",
                    #    "instance": Speed()
                    #     },
                   ##  "dfc": {
                   #      "desc": "Distance from flock center",
                   #      "unit": "m",
                   #      "axis_label": "Distance from flock center",
                   #      "instance": distfromRT()
                   #     },     
                    #"TPtraj": {
                    #     "desc": "Difference from optimal trajectory",
                    #     "unit": "$^\circ$",
                    #     "axis_label": "Angle From Optimal Trajectory",
                    #     "instance": TPTrajectoryMetric()
                    #    },
                    # "distfromRT": {
                    #      "desc": "Distance from Racetrack",
                    #      "unit": "m",
                    #      "axis_label": "Distance from Racetrack",
                    #      "instance": distfromRT()
                    #     },
                    #"distfromCircle": {
                    #     "desc": "Distance from Circle",
                    #     "unit": "m",
                    #     "axis_label": "Distance from Circle",
                    #     "instance": circleCentreDist()
                    #    },
                    #"FHtraj" : {
                    #    "desc": "Difference from optimal trajectory",
                    #    "unit": "$^\circ$",
                    #    "axis_label": "Angle from Optimal trajectory",
                    #    "instance": FHTrajectoryMetric()
                    #}
    }