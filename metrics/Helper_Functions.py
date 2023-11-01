import pandas as pd
import math

class HelperFunctions():
 
    

    def getOrientations(self, data: pd.DataFrame) -> pd.DataFrame:
        xvels = data.iloc[:, 1]
        yvels = data.iloc[:, 2]
        bearing = 0
        bearings = []
        for x in range(len(xvels)):
            if int(yvels[x] )== 0 or int(xvels[x]) == 0:
                angle = 0
            else:
                angle = math.atan(int(yvels[x]) / int(xvels[x]))

            if int(xvels[x]) < 0 and int(yvels[x]) < 0:
                bearing = 180 + (90 - angle)
            elif int(xvels[x]) < 0 and int(yvels[x]) >= 0:  
                bearing = 270 + angle
            elif int(xvels[x]) >= 0 and int(yvels[x]) < 0:
                bearing = (90 + angle)
            elif int(xvels[x]) >= 0 and int(yvels[x]) >= 0:
                bearing = (90 - angle)
            bearings.append(bearing)
        return bearings
    