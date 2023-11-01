import time
import wx
from controller import Controller
from flock_settings import Setting
from wx.lib.intctrl import IntCtrl


class SimulatorSettingsPanel(wx.Panel):

    def __init__(self, parent, controller : Controller):
        wx.Panel.__init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(
            300, 300), style=wx.TAB_TRAVERSAL)

        self.controller = controller

        sizer = wx.GridBagSizer(hgap=5, vgap=5)
        sizer_h = wx.BoxSizer(wx.VERTICAL)

        self.text_Blank1 = wx.StaticText(
            self, label="", style=wx.SL_HORIZONTAL)
        self.text_Blank2 = wx.StaticText(
            self, label="", style=wx.SL_HORIZONTAL)
        self.text_Blank3 = wx.StaticText(
            self, label="", style=wx.SL_HORIZONTAL)
        self.text_Blank4 = wx.StaticText(
            self, label="", style=wx.SL_HORIZONTAL)
        self.text_Blank5 = wx.StaticText(
            self, label="", style=wx.SL_HORIZONTAL)

        self.text_Simulation = wx.StaticText(
            self, label="Reset Simulation for changed drone properties to take effect", style=wx.SL_HORIZONTAL)
        self.text_Simulation.Wrap(100)

        self.button_Reset = wx.Button(
            self, wx.ID_ANY, u"Reset Simulation", wx.DefaultPosition, wx.DefaultSize, 0)

        self.text_seed = wx.StaticText(
            self, label="Seed", style=wx.ALIGN_CENTRE)
        self.text_box_seed = wx.TextCtrl(self, value="")

        self.checkbox_dynamicView = wx.CheckBox(self, label="Dynamic View")

        self.text_boundsDisplay = wx.StaticText(
            self, label="Display Bounds", style=wx.ALIGN_CENTRE)
        self.text_maxBounds = wx.StaticText(self, label="Max ({0}, {1})".format(
            controller.get_preview_max_x(), controller.get_preview_max_y()), style=wx.ALIGN_CENTRE)

        self.text_X = wx.StaticText(
            self, label="X Bounds", style=wx.ALIGN_CENTRE)
        self.text_box_boundsDisplayXLower = IntCtrl(self, value=controller.get_preview_min_x())
        self.text_box_boundsDisplayYLower = IntCtrl(self, value=controller.get_preview_min_y())

        self.text_Y = wx.StaticText(
            self, label="Y Bounds", style=wx.ALIGN_CENTRE)
        self.text_box_boundsDisplayXUpper = IntCtrl(self, value=controller.get_preview_max_x())
        self.text_box_boundsDisplayYUpper = IntCtrl(self, value=controller.get_preview_max_y())

        self.text_Lower_1 = wx.StaticText(self, label="Lower")
        self.text_Lower_2 = wx.StaticText(self, label="Lower")
        self.text_Upper_1 = wx.StaticText(self, label="Upper")
        self.text_Upper_2 = wx.StaticText(self, label="Upper")

        self.checkbox_comms_ranges = wx.CheckBox(self, label="Show Comms Ranges")
        self.checkbox_message_propagation = wx.CheckBox(self, label="Show Message Propagation (speed 1 only)")
        self.checkbox_position_approximation = wx.CheckBox(self, label="Show Position Approximation")
        self.checkbox_neighbour_approximation = wx.CheckBox(self, label="Show Neighbour Position Approximations")

        self.text_TimeStep = wx.StaticText(
            self, label="Timestep: ", style=wx.ALIGN_CENTRE)

        default_speed = self.controller.sim_speed

        self.text_simulationSpeed = wx.StaticText(
            self, label="Simulation Speed")
        self.update_speed_label(default_speed)
        self.slider_simulationSpeed = wx.Slider(
            self, value=default_speed, minValue=0, maxValue=8, style=wx.SL_LABELS | wx.SL_AUTOTICKS, size=(300, 100))
        self.slider_simulationSpeed.SetTickFreq(1)

        sizer.Add(self.text_Blank1, pos=(0, 0), border=20)

        sizer.Add(self.button_Reset, pos=(1, 0), border=20)
        sizer.Add(self.text_Simulation, pos=(2, 0), border=20)
        sizer.Add(self.text_seed, pos=(3, 0), border=20)
        sizer.Add(self.text_box_seed, pos=(4, 0), border=20)

        sizer.Add(self.text_Blank2, pos=(5, 0), border=20)

        sizer.Add(self.checkbox_dynamicView, pos=(6, 0), border=50)

        sizer.Add(self.text_boundsDisplay, pos=(7, 0), border=20)
        sizer.Add(self.text_maxBounds, pos=(8, 0), border=20)

        sizer.Add(self.text_Blank3, pos=(9, 0), border=20)
        sizer.Add(self.text_X, pos=(10, 0), border=20)
        sizer.Add(self.text_Lower_1, pos=(11, 0), border=20)
        sizer.Add(self.text_Upper_1, pos=(11, 1), border=20)
        sizer.Add(self.text_box_boundsDisplayXLower, pos=(12, 0), border=20)
        sizer.Add(self.text_box_boundsDisplayXUpper, pos=(12, 1), border=20)

        sizer.Add(self.text_Blank4, pos=(13, 0), border=20)
        sizer.Add(self.text_Y, pos=(14, 0), border=20)
        sizer.Add(self.text_Lower_2, pos=(15, 0), border=20)
        sizer.Add(self.text_Upper_2, pos=(15, 1), border=20)
        sizer.Add(self.text_box_boundsDisplayYLower, pos=(16, 0), border=20)
        sizer.Add(self.text_box_boundsDisplayYUpper, pos=(16, 1), border=20)

        sizer.Add(self.text_Blank5, pos=(17, 0), border=20)
        
        sizer.Add(self.checkbox_comms_ranges, pos=(18,0), span=(1, 2), border = 20)
        sizer.Add(self.checkbox_message_propagation, pos=(19,0), span=(1, 2), border = 20)
        sizer.Add(self.checkbox_position_approximation, pos=(20,0), span=(1, 2), border = 20)
        sizer.Add(self.checkbox_neighbour_approximation, pos=(21,0), span=(1, 2), border = 20)
        
        sizer.Add(self.text_TimeStep, pos=(22, 0), border=20)
        sizer.Add(self.text_simulationSpeed, pos=(23, 0), border=20)
        sizer.Add(self.slider_simulationSpeed,
                  pos=(24, 0), span=(1, 2), border=20)

        sizer_h.Add(sizer, 1, wx.CENTER)

        self.SetSizer(sizer_h)
        self.Layout()

        self.Bind(wx.EVT_BUTTON, self.on_click_reset_simulation,
                  self.button_Reset)

        self.Bind(wx.EVT_TEXT, self.on_seed_change, self.text_box_seed)

        self.Bind(wx.EVT_TEXT, self.on_text_change_x_lower,
                  self.text_box_boundsDisplayXLower)
        self.Bind(wx.EVT_TEXT, self.on_text_change_y_lower,
                  self.text_box_boundsDisplayYLower)
        self.Bind(wx.EVT_TEXT, self.on_text_change_x_upper,
                  self.text_box_boundsDisplayXUpper)
        self.Bind(wx.EVT_TEXT, self.on_text_change_y_upper,
                  self.text_box_boundsDisplayYUpper)
        self.Bind(wx.EVT_CHECKBOX, self.on_dynamic_view_toggled,
                  self.checkbox_dynamicView)
        
        self.Bind(wx.EVT_CHECKBOX, 
                  self.on_show_comms_ranges_toggled,
                  self.checkbox_comms_ranges)
        
        self.Bind(wx.EVT_CHECKBOX, 
                  self.on_show_message_propagation_toggled,
                  self.checkbox_message_propagation)
        
        self.Bind(wx.EVT_CHECKBOX, 
                  self.on_show_position_approximation_toggled,
                  self.checkbox_position_approximation)
        
        self.Bind(wx.EVT_CHECKBOX, 
                  self.on_show_neighbour_approximations_toggled,
                  self.checkbox_neighbour_approximation)
        
        self.Bind(wx.EVT_SLIDER, self.on_simulation_speed_changed,
                  self.slider_simulationSpeed)

        self.checkbox_dynamicView.SetValue(True)
        self.set_manual_bounds_controls_enabled(False)

    def update_time(self, timestep):
        self.text_TimeStep.SetLabel("Timestep: " + str(timestep))

    def on_click_reset_simulation(self, event):
        if(self.text_box_seed.GetValue() == ""):
            self.controller.settings.set(Setting.SEED, int(time.time()))

        self.controller.reset()

    def on_seed_change(self, event):
        string = self.text_box_seed.GetValue()

        if string.isnumeric():
            self.controller.settings.set(Setting.SEED, int(string))
        else:
            self.text_box_seed.SetValue("")

    def on_text_change_x_lower(self, event):
        self.text_box_boundsDisplayXLower.SetValue(
            self.controller.try_set_preview_min_x(
                self.text_box_boundsDisplayXLower.GetValue()))

    def on_text_change_y_lower(self, event):
        self.text_box_boundsDisplayYLower.SetValue(
            self.controller.try_set_preview_min_y(
                self.text_box_boundsDisplayYLower.GetValue()))

    def on_text_change_x_upper(self, event):
        self.text_box_boundsDisplayXUpper.SetValue(
            self.controller.try_set_preview_max_x(
                self.text_box_boundsDisplayXUpper.GetValue()))

    def on_text_change_y_upper(self, event):
        self.text_box_boundsDisplayYUpper.SetValue(
            self.controller.try_set_preview_max_y(
                self.text_box_boundsDisplayYUpper.GetValue()))

    def on_dynamic_view_toggled(self, event):
        isChecked = self.checkbox_dynamicView.IsChecked()

        self.set_manual_bounds_controls_enabled(not isChecked)

        self.controller.dynamic_view_enabled  = isChecked
        
    def on_show_comms_ranges_toggled(self, event):
        self.controller.show_comms_ranges = self.checkbox_comms_ranges.IsChecked()
        
    def on_show_message_propagation_toggled(self, event):
        self.controller.show_message_propagtion = self.checkbox_message_propagation.IsChecked()
        
    def on_show_position_approximation_toggled(self, event):
        self.controller.show_position_approximation = self.checkbox_position_approximation.IsChecked()
        
    def on_show_neighbour_approximations_toggled(self, event):
        self.controller.show_neigbour_approximations = self.checkbox_neighbour_approximation.IsChecked()

    def set_manual_bounds_controls_enabled(self, value):
        self.text_maxBounds.Enabled = value
        self.text_boundsDisplay.Enabled = value

        self.text_X.Enabled = value
        self.text_Y.Enabled = value
        self.text_Upper_1.Enabled = value
        self.text_Upper_2.Enabled = value
        self.text_Lower_1.Enabled = value
        self.text_Lower_2.Enabled = value

        self.text_box_boundsDisplayXLower.Enabled = value
        self.text_box_boundsDisplayYLower.Enabled = value
        self.text_box_boundsDisplayXUpper.Enabled = value
        self.text_box_boundsDisplayYUpper.Enabled = value

    def on_simulation_speed_changed(self, event):
        value = self.slider_simulationSpeed.GetValue()
        if(value > 0):
            speed = 2**(value-1)
        else:
            speed = 0
        self.controller.sim_speed = speed
        self.update_speed_label(speed)

    def update_speed_label(self, value):
        self.text_simulationSpeed.SetLabel(
            "Simulation Speed (x{})".format(value))
