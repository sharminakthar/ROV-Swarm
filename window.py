# %matplotlib auto
import wx
from matplotlib import *
import matplotlib.pyplot as plt
from wx.lib.agw import aui
from gui.flock_settings_panel import FlockSettingsPanel
from gui.drone_preview_panel import DronePreviewPanel
from gui.drone_readings_panel import DroneReadingsPanel
from gui.main_frame import MainFrame
from gui.metrics_panel import MetricsPanel
from gui.simulator_settings_panel import SimulatorSettingsPanel
from controller import Controller


# Main Application that contains all of the window panels
class MainApp(MainFrame):
    def __init__(self, parent):
        MainFrame.__init__(self, parent)
        self.Bind(wx.EVT_CLOSE, self.__OnQuit)
        self.SetMinSize((1280, 720))

        self.controller = Controller()

        self._manager = aui.AuiManager()
        self._manager.SetManagedWindow(self)

        self.panel = wx.Panel()

        self.drone_readings_panel = DroneReadingsPanel(
            parent=self, controller=self.controller)
        self.drone_readings_panel.SetMinSize((400, 400))
        drone_readings_panel_info = aui.AuiPaneInfo().Name('DroneReadingsPanel').Caption('Drone Readings').Left().\
            Show().Floatable(False).CloseButton(False).Movable(False).MinSize((200, 100))

        self.metrics_panel = MetricsPanel(
            parent=self, controller=self.controller)
        self.metrics_panel.SetMinSize((400, 400))
        metrics_panel_info = aui.AuiPaneInfo().Name('MetricsPanel').Caption('Metrics Panel').Left().\
            Show().Floatable(False).CloseButton(False).Movable(False).MinSize((200, 600))

        self.flock_settings_panel = FlockSettingsPanel(
            parent=self, controller=self.controller)
        self.flock_settings_panel.SetMinSize((200, 200))
        flock_settings_panel_info = aui.AuiPaneInfo().Name('FlockSettingsPanel').Caption('Flock Settings').Left().Row(1).\
            Show().Floatable(False).CloseButton(False).Movable(False).MinSize((300, 100))

        self.simulator_settings_panel = SimulatorSettingsPanel(
            parent=self, controller=self.controller)
        self.simulator_settings_panel.SetMinSize((300, 200))
        simulator_settings_panel_info = aui.AuiPaneInfo().Name('SimulatorSettingsPanel').Caption('Simulator Settings').Right(). \
            Show().Floatable(False).CloseButton(False).Movable(False).MinSize((200, 100))

        self.preview_panel = DronePreviewPanel(
            parent=self, controller=self.controller)
        self.preview_panel.SetMinSize((1000, 1000))
        preview_panel_Info = aui.AuiPaneInfo().Name('SimulationPanel').Caption('Simulation').Center(). \
            Show().Floatable(False).CloseButton(False).Movable(False).MinSize((200, 100))

        self._manager.AddPane(self.drone_readings_panel,
                              drone_readings_panel_info)
        self._manager.AddPane(self.metrics_panel, metrics_panel_info)
        self._manager.AddPane(self.flock_settings_panel,
                              flock_settings_panel_info)
        self._manager.AddPane(self.simulator_settings_panel,
                              simulator_settings_panel_info)
        self._manager.AddPane(self.preview_panel, preview_panel_Info)

        self._manager.Update()

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        # attempt to refresh every 1ms, will fallback to going as fast as possible
        self.timer.Start(1)

    def update(self, event):
        self.controller.update()
        self.drone_readings_panel.update_values()
        self.metrics_panel.update_values()
        self.preview_panel.update()
        self.simulator_settings_panel.update_time(
            self.controller.simulator.get_step())

    def __OnQuit(self, event):
        self.timer.Stop()
        plt.close()
        del self._manager
        self.Destroy()


app = wx.App()
window = MainApp(None)

window.Show(True)
app.MainLoop()
