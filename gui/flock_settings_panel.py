import wx
import wx.lib.scrolledpanel as scroll
from controller import Controller
from flock_settings import Setting


class FlockSettingsPanel(scroll.ScrolledPanel):

    def __init__(self, parent, controller : Controller):
        wx.Panel.__init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(
            300, 300), style=wx.SL_HORIZONTAL | wx.SL_LABELS)

        self.controller = controller

        self.sizer = wx.GridBagSizer(hgap=5, vgap=6)
        self.sizer_h = wx.BoxSizer(wx.VERTICAL)

        self.resets = []

        self.button_Default = wx.Button(
            self, wx.ID_ANY, u"Reset to Default", wx.DefaultPosition, wx.DefaultSize, 0)

        self.sizer.Add(self.button_Default, pos=(1, 0), span=(1,2), flag=wx.ALIGN_CENTRE_HORIZONTAL)
        self.Bind(wx.EVT_BUTTON, self.reset, self.button_Default)

        self.inserstion_pos = 2

        self.add_heading("Flock")

        self.add_input("Drone Number", 1, 20, 1, Setting.FLOCK_SIZE)
        self.add_input("Separation Distance (m)", 5, 100, 1, Setting.SEPARATION_DISTANCE)

        self.add_heading("Physics")

        self.add_input("Max Speed (ms^-1)", 0, 100, .5, Setting.MAX_SPEED)
        self.add_input("Max Acceleration (ms^-2)", 0, 100, .5, Setting.MAX_ACCELERATION)
        self.add_input("Max Deceleration (ms^-2)", 0, 100, .5, Setting.MAX_DECELERATION)
        self.add_input("Max Rate of Turn (deg/s)", 0, 360, .5, Setting.MAX_RATE_OF_TURN)
        self.add_input("Mothership Max Speed (ms^-1)", 0, 100, .5, Setting.MOTHERSHIP_MAX_SPEED)

        self.add_heading("Communications")

        self.add_input("Max Range (m)", 500, 3000, 100, Setting.MAX_RANGE)
        self.add_input("Bandwidth (B/s)", 1, 24, 1, Setting.BANDWIDTH)
        self.add_input("Message Size (B)", 4, 24, 1, Setting.MESSAGE_SIZE)
        self.add_input("Packet loss (%)", 0, 100, 1, Setting.PACKET_LOSS)
        
        self.add_heading("Sensor Noise Errors (SD)")

        self.add_input("Speed Error (%)", 0, 30, 1, Setting.SPEED_ERROR)
        self.add_input("Heading Error (deg)", 0, 30, 0.1, Setting.HEADING_ERROR)
        self.add_input("Relative Range Error (%)", 0, 10, 5, Setting.RANGE_ERROR)
        self.add_input("Relative Bearing Error (deg)", 0, 15, .5, Setting.BEARING_ERROR)
        self.add_input("Acceleration Error (%)", 0, 30, .5, Setting.ACCELERATION_ERROR)

        self.add_heading("Calibration Errors (SD)")

        self.add_input("Speed Calibration Error (ms^-1)", 0, 1, 0.01, Setting.SPEED_CALIBRATION_ERROR)
        self.add_input("Heading Calibration Error (deg)", 0, 30, 0.1, Setting.HEADING_CALIBRATION_ERROR)
        self.add_input("Range Calibration Error (m)", 0, 30, .5, Setting.RANGE_CALIBRATION_ERROR)
        self.add_input("Bearing Calibration Error (deg)", 0, 30, 0.1, Setting.BEARING_CALIBRATION_ERROR)
        self.add_input("Acc. Calibration Error (ms^-2)", 0, 1, 0.01, Setting.ACCELERATION_CALIBRATION_ERROR)
        
        self.add_heading("Objectives")
        
        self.add_enum_input("Drone Objective", Setting.DRONE_OBJECTIVE)
        self.add_enum_input("Mothership Objective", Setting.MOTHERSHIP_OBJECTIVE)

        self.add_input("Target X", -10000, 10000, 100, Setting.TARGET_X)
        self.add_input("Target Y", -10000, 10000, 100, Setting.TARGET_X)
        self.add_input("Target Heading", 0, 360, 5, Setting.TARGET_HEADING)
        self.add_input("Target Radius", 100, 10000, 100, Setting.TARGET_RADIUS)  

        self.add_heading("Algorithm Weights")
        self.add_input("Separation Vector", 0, 2, 0.01, Setting.WEIGHT_SEPARATION)  
        self.add_input("Alignment Vector", 0, 2, 0.01, Setting.WEIGHT_ALIGNMENT) 
        self.add_input("Cohesion Vector", 0, 2, 0.01, Setting.WEIGHT_COHESION) 
        self.add_input("Objective Vector", 0, 2, 0.01, Setting.WEIGHT_OBJECTIVE)     

        self.sizer_h.Add(self.sizer, 1, wx.CENTER)

        self.SetSizer(self.sizer_h)

        self.SetupScrolling()

    def add_input(self, label, min_value, max_value, increments, setting):
        text = wx.StaticText(self, label=label, style=wx.SL_HORIZONTAL)

        value = self.controller.settings.get(setting)

        if(increments == 1):
            spin_ctrl = wx.SpinCtrl(self, initial=value, min=min_value, max=max_value, style=wx.SL_VERTICAL | wx.SP_ARROW_KEYS)
            event = wx.EVT_SPINCTRL
        else:
            spin_ctrl = wx.SpinCtrlDouble(self, initial=value, min=min_value, max=max_value, inc=increments, style=wx.SL_VERTICAL | wx.SP_ARROW_KEYS)
            event = wx.EVT_SPINCTRLDOUBLE

        size = spin_ctrl.GetSizeFromTextSize(spin_ctrl.GetTextExtent('0000000000'))
        spin_ctrl.SetMinSize(size)
        spin_ctrl.SetMaxSize(size)

        self.sizer.Add(text, pos=(self.inserstion_pos, 0), flag=wx.ALIGN_LEFT)
        self.sizer.Add(spin_ctrl, pos=(self.inserstion_pos, 1), flag=wx.ALIGN_LEFT)

        self.inserstion_pos += 1

        self.Bind(event, lambda event : self.controller.settings.set(setting, event.GetEventObject().GetValue()), spin_ctrl) 
        
        self.resets.append(lambda : self.controller.settings.reset(setting)) 
        self.resets.append(lambda : spin_ctrl.SetValue(self.controller.settings.get(setting)))   

    def add_enum_input(self, label, setting):
        text = wx.StaticText(self, label=label, style=wx.SL_HORIZONTAL)

        default = self.controller.settings.get(setting)
        type_of = type(default)

        choices = list(map(lambda s : s.lower(), type_of._member_names_))

        combo_box = wx.ComboBox(self, choices=choices, value=default.name.lower())

        self.sizer.Add(text, pos=(self.inserstion_pos, 0), flag=wx.ALIGN_LEFT)
        self.sizer.Add(combo_box, pos=(self.inserstion_pos, 1), flag=wx.ALIGN_LEFT)

        self.inserstion_pos += 1

        size = combo_box.GetSizeFromTextSize(combo_box.GetTextExtent('0000000000'))
        combo_box.SetMinSize(size)
        combo_box.SetMaxSize(size)

        font = wx.Font(8, wx.DEFAULT, wx.ITALIC, wx.NORMAL, underline=False)

        combo_box.SetFont(font)

        self.Bind(wx.EVT_COMBOBOX, lambda event : self.controller.settings.set(setting, type_of[event.GetEventObject().GetValue().upper()]), combo_box) 

        self.resets.append(lambda : self.controller.settings.reset(setting)) 
        self.resets.append(lambda : combo_box.SetValue(self.controller.settings.get(setting).name.lower()))   

    def reset (self, event):
        for resetFunction in self.resets:
            resetFunction()

    def add_heading(self, label):
        text = wx.StaticText(self, label=label, style=wx.SL_HORIZONTAL)

        font = wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD, underline=True)

        text.SetFont(font)
        self.sizer.Add(1, 1, pos=(self.inserstion_pos,0))
        self.sizer.Add(text, pos=(self.inserstion_pos+1, 0), span=(1,2), flag=wx.ALIGN_CENTER)

        self.inserstion_pos+=2
