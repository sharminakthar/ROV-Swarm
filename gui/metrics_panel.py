import wx
from controller import Controller
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg


class MetricsPanel(wx.Panel):

    def __init__(self, parent, controller : Controller):
        wx.Panel.__init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(
            100, 100), style=wx.TAB_TRAVERSAL)

        self.controller = controller

        self.list_ctrl = wx.ListCtrl(self, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        
        self.plot = MetricsPlotter(self, controller)

        sizer = wx.BoxSizer(wx.VERTICAL)        
        
        sizer.Add(self.list_ctrl, 1, wx.EXPAND, 5) 
        sizer.Add(self.plot, 2, wx.EXPAND, 10)

        self.SetSizer(sizer)
        
        self.update_metrics_definitions()

        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.onItemSelected, self.list_ctrl) 

    def update_metrics_definitions(self):
        self.list_ctrl.ClearAll()
        self.list_ctrl.InsertColumn(0, 'Metric')
        self.list_ctrl.InsertColumn(1, 'Value')
        self.list_ctrl.SetColumnWidth(0, 200)
        self.list_ctrl.SetColumnWidth(1, 200)
        self.definitions = self.controller.simulator.metric_definitions
        for i, name in enumerate(self.definitions):        
            self.list_ctrl.InsertItem(i,name)

    # Gets the values recieved from the controller.
    def update_values(self):
        if(not (self.definitions is self.controller.simulator.metric_definitions)):            
            self.update_metrics_definitions()
        for i, (name, function) in enumerate(self.controller.simulator.metric_definitions.items()):
            format = self.controller.simulator.metric_formats[name]            
            self.list_ctrl.SetItem(i,1,format(function()))
        self.plot.update_values()

    def onItemSelected(self, event):
        index = self.list_ctrl.GetFirstSelected()
        self.plot.select_metric(index)

class MetricsPlotter(wx.Panel):
    def __init__(self, parent, controller : Controller):
        wx.Panel.__init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(
            100, 100), style=wx.TAB_TRAVERSAL)

        self.controller = controller
        
        self.figure = mpl.figure.Figure(figsize=(1, 1), dpi=80)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(self.sizer)

        self.selection = 1

    def update_values(self):        
        self.axes.clear()
        self.controller.simulator.get_metrics_log().plot(ax=self.axes, x="Timestep",y=self.selection)
        self.figure.canvas.draw()        

    def select_metric(self, value):
        self.selection = value + 1