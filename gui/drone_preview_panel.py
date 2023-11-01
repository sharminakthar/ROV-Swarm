import wx
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.patches import Circle, Rectangle, ConnectionPatch
import numpy as np
import colorsys
from controller import Controller
from flock_settings import Setting
from simulation.flock import Flock

from simulation.objectives import DroneObjective, MothershipObjective

class DronePreviewPanel(wx.Panel):
    def __init__(self, parent, controller : Controller):
        wx.Panel.__init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(
            100, 100), style=wx.TAB_TRAVERSAL)

        self.controller = controller
        self.figure = mpl.figure.Figure()

        self.axes = self.figure.add_subplot(111)
        self.axes.autoscale(enable=False)
        self.axes.set_xlabel("X (m)")
        self.axes.set_ylabel("Y (m)")

        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(self.sizer)

        self.selection = 1  

        self.dictionary = {}
        self.dictionaryLabel = {}
        self.dictionaryLine = {}
        self.mothershipDisplay = []
        #self.Fit()
        self.simulationStep = 0
        self.initalized = False

    def update(self):        
        self.axes.clear()
        
        flock = self.controller.simulator.get_flock()

        if(self.controller.show_message_propagtion):
            self.draw_message_visualisation(flock)
            
        if(self.controller.show_comms_ranges):
            self.draw_comms_ranges(flock)

        if(self.controller.show_neigbour_approximations):
            self.draw_neighbour_approximations(flock)

        for i in range(0, flock.get_size()):
            self.draw_drone(flock, i)

        self.draw_mission_specifics()

        self.set_axis_limits(flock)

        self.figure.canvas.draw()
        
    def draw_drone(self, flock : Flock, id):        
        x = flock.get_positions()[0][id]
        y = flock.get_positions()[1][id]

        drone = flock.get_drone(id)
        
        angle = drone.get_heading()

        approx_position = drone.get_approximated_position().flatten()

        color = self.get_marker_color(flock, id)

        label = id
        drone = flock.get_drone(id)
        if(drone.is_mothership()):
            label = str(id) + ' (M)'

        self.axes.plot(x,y, marker=(3, 0, -angle), ms=10, ls="", markerfacecolor=color, markeredgecolor=color)
        self.axes.annotate(label, xy=(x + 10, y + 10), xytext=(x + 10, y + 10), annotation_clip=False)
        self.axes.plot(x, y, marker=(2, 0, -angle), ms=10, ls="", markerfacecolor=color, markeredgecolor=color)

        if(self.controller.show_position_approximation):
            self.axes.plot(approx_position[0], approx_position[1], marker="x", ms=5, ls="", markerfacecolor=color, markeredgecolor=color)       
        
    def draw_neighbour_approximations(self, flock: Flock):
        for drone in flock.drones:
            for neighbour_info in drone.drone_controller.flock_info:
                if(neighbour_info is None):
                    continue
                color_1 = self.get_marker_color(flock, drone.my_id)
                color_2 = self.get_marker_color(flock, neighbour_info.get_drone_id())
                
                self_pos_error = drone.get_exact_position() - drone.get_approximated_position()
                
                pos = (neighbour_info.get_position() + self_pos_error).flatten()
                self.axes.plot(pos[0], pos[1], marker="o", ms=5, ls="", markerfacecolor="white", markeredgecolor=color_1)
                self.axes.plot(pos[0], pos[1], marker="+", ms=5, ls="", markerfacecolor=None, markeredgecolor=color_2)

    def get_marker_color(self, flock, drone_id, s=.45, v=.9, mothership_value=0):
        drone = flock.get_drone(drone_id)

        if(drone.is_mothership()):
            return (mothership_value,mothership_value,mothership_value)

        h = drone_id / flock.get_size()

        return colorsys.hsv_to_rgb(h,s,v)

    def set_axis_limits(self, flock):
        self.axes.set_aspect(1)

        if(self.controller.dynamic_view_enabled):
            min_x = min(flock.get_positions()[0])
            max_x = max(flock.get_positions()[0])
            min_y = min(flock.get_positions()[1])
            max_y = max(flock.get_positions()[1])

            width = max_x - min_x
            height = max_y - min_y

            width = max(width, height)
            height = width

            mid_x = (max_x + min_x) / 2
            mid_y = (max_y + min_y) / 2

            x_lim_lower = mid_x - width / 2 - 100
            x_lim_upper = mid_x + width / 2 + 100

            y_lim_lower = mid_y - width / 2 - 100
            y_lim_upper = mid_y + width / 2 + 100
            
            self.axes.set_xlim(x_lim_lower, x_lim_upper)
            self.axes.set_ylim(y_lim_lower, y_lim_upper)
        else:
            self.axes.set_xlim(self.controller.get_preview_min_x(), self.controller.get_preview_max_x())
            self.axes.set_ylim(self.controller.get_preview_min_y(), self.controller.get_preview_max_y())

        self.axes.set_xlabel("X (m)")
        self.axes.set_ylabel("Y (m)")

    def draw_message_visualisation(self, flock : Flock):
        if(self.controller.sim_speed > 1):
            return

        messages = flock.get_messages_sent_last_step()

        for message in messages:
            sender = message.get_drone_id()

            posA = flock.get_drone(sender).get_exact_position()

            for receiver in range(0, flock.get_size()):
                if(flock.drones_in_range(sender, receiver)):
                    posB = flock.get_drone(receiver).get_exact_position()

                    connect = ConnectionPatch(posA, posB, "data",
                      arrowstyle="-|>", shrinkA=15, shrinkB=15, 
                      mutation_scale=20,
                      color=self.get_marker_color(flock, sender, s=.2, v=1, mothership_value=0.8))

                    self.axes.add_patch(connect)

           
    def draw_comms_ranges(self, flock : Flock):
        com_range = self.controller.settings.get_cached(Setting.MAX_RANGE)
        
        for i in range(0, flock.get_size()):
            pos = flock.get_drone(i).get_exact_position()

            circle = Circle(pos, com_range, fill=False, linewidth=0.5, 
            linestyle='-', color=self.get_marker_color(flock, i, s=.2, v=1, mothership_value=0.8))
            self.axes.add_patch(circle) 




    def draw_mission_specifics(self):
        settings = self.controller.settings

        drone_objective = settings.get_cached(Setting.DRONE_OBJECTIVE)
        mother_objective = settings.get_cached(Setting.MOTHERSHIP_OBJECTIVE)
        pos = np.array([settings.get_cached(Setting.TARGET_X), settings.get(Setting.TARGET_Y)])
        radius = settings.get_cached(Setting.TARGET_RADIUS)
        heading = settings.get_cached(Setting.TARGET_HEADING)

        if(drone_objective == DroneObjective.FOLLOW_CIRCLE 
                or mother_objective == MothershipObjective.FOLLOW_CIRCLE):
            circle = Circle(pos, radius, fill=False, linewidth=0.5, linestyle='--')
            self.axes.add_patch(circle)
        elif (drone_objective == DroneObjective.TARGET_POINT 
                or mother_objective == MothershipObjective.TARGET_POINT):
            circle = Circle(pos, 100, fill=False, linewidth=0.5, linestyle='--')
            self.axes.add_patch(circle)
        elif (drone_objective == DroneObjective.FIXED_HEADING):
            rect = Rectangle([0,0],10,100000,angle=-heading, fill=False, linewidth=0.5, linestyle='--')
            self.axes.add_patch(rect)
        elif (drone_objective == DroneObjective.RACETRACK):
            rect = Rectangle([1500,1500], 2000,2000)
            circle1 = Circle([3500,2500], 1000)
            circle2 = Circle([1500,2500], 1000)

            self.axes.add_patch(rect)
            self.axes.add_patch(circle1)
            self.axes.add_patch(circle2)



        
