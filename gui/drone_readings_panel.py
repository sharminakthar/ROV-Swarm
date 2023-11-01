import wx
from wx import grid
import numpy as np
from controller import Controller


class DroneReadingsPanel(wx.Panel):

    def __init__(self, parent, controller: Controller):
        wx.Panel.__init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(
            800, 400), style=wx.TAB_TRAVERSAL)

        self.controller = controller

        droneNum = self.controller.simulator.get_flock().get_size()

        # Creates Grid for the data table

        self.myGrid = grid.Grid(self)
        self.myGrid.CreateGrid(droneNum, 5)

        self.myGrid.EnableEditing(False)
        self.myGrid.SetRowLabelSize(0)
        self.myGrid.SetColLabelValue(0, "Drone ID")
        self.myGrid.SetColLabelValue(1, "X")
        self.myGrid.SetColLabelValue(2, "Y")
        self.myGrid.SetColLabelValue(3, "Heading (°N)")
        self.myGrid.SetColLabelValue(4, "Speed (m/s)")
        self.previousDroneNum = droneNum

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.myGrid, 1, wx.EXPAND)
        self.SetSizer(sizer)

    # Gets the values recieved from the controller.
    def update_values(self):
        droneNum = self.controller.simulator.get_flock().get_size()
        rowNumberDifference = droneNum - self.myGrid.GetNumberRows()

        if rowNumberDifference > 0:
            self.myGrid.AppendRows(rowNumberDifference)
        elif rowNumberDifference < 0:
            self.myGrid.DeleteRows(
                pos=self.myGrid.GetNumberRows() - abs(rowNumberDifference), numRows=abs(rowNumberDifference))

        self.previousDroneNum = droneNum

        for i in range(0, droneNum):
            self.myGrid.SetCellValue(i, 0, str(i))

        flock = self.controller.simulator.get_flock()

        for i in range(0, droneNum):
            self.myGrid.SetCellValue(
                i, 1, str(round(flock.get_positions()[0][i], 3)))
            self.myGrid.SetCellValue(
                i, 2, str(round(flock.get_positions()[1][i], 3)))
            self.myGrid.SetCellValue(
                i, 3, str(round(flock.get_headings()[i], 3)))
            self.myGrid.SetCellValue(
                i, 4, str(round(flock.get_speeds()[i], 3)))
