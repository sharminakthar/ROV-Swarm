from pathlib import Path
from BaseMetric import BaseMetric
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import os 

def GetHeatmap(mission,para,run,bins,path):
    pathname = "../out/"+mission+'/'+para
    filenames = os.listdir(pathname)
    for x in range(len(filenames)):
        if filenames[x].isdigit():
            filenames[x] = int(filenames[x])
        else:
            filenames[x] = float(filenames[x])
    filenames = sorted(filenames)
    del filenames[1]
    filenames = [str(x) for x in filenames]

    a = int(len(filenames)/3)

    fig, axes = plt.subplots(a, 3 ,figsize = [8,8])

    for i in range(a):
        for j in range(3):
            data0 = pd.read_csv(pathname+'/'+filenames[(i*3)+j]+'/'+run+"/raw_data_log.csv")
            x_p0 = data0["X Position"]
            y_p0 = data0["Y Position"]
            x_array0 = x_p0.to_numpy()
            y_array0 = y_p0.to_numpy()

            heatmap0, xedges0, yedges0 = np.histogram2d(x_array0, y_array0, bins=bins)
            minp = min(xedges0[0],xedges0[-1],yedges0[0],yedges0[-1])
            maxp = max(xedges0[0],xedges0[-1],yedges0[0],yedges0[-1])
            extent0 = [minp,maxp ,minp , maxp]
            x_points = np.linspace(start = minp, stop = maxp, num = 3)

            axes[i][j].imshow(heatmap0.T, cmap = 'turbo', extent=extent0, origin='lower')
            axes[i][j].set_title(filenames[(i*3)+j],fontdict={'fontsize': 12})
            axes[i][j].xaxis.set_major_locator(FixedLocator([round(item, -2) for item in x_points]))
            if path:
                axes[i][j].add_patch(Circle((2500,2500),1000, fill = False, color = '#39FF14'))

    fig.suptitle(str.title(para.replace("_", " " )), fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    bins = 200
    run = '3'
    mission = {
        '1' : 'FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA',
        '2' : 'FIXED_HEADING'
    }

    parameters = {
        '1' : 'ACCELERATION_CALIBRATION_ERROR',
        '2' : 'ACCELERATION_ERROR',
        '3' : 'BANDWIDTH',
        '4' : 'BEARING_CALIBRATION_ERROR',
        '5' : 'BEARING_ERROR',
        '6' : 'FLOCK_SIZE',
        '7' : 'HEADING_CALIBRATION_ERROR',
        '8' : 'HEADING_ERROR',
        '9' : 'PACKET_LOSS',
        '10' : 'RANGE_CALIBRATION_ERROR',
        '11' : 'RANGE_ERROR',
        '12' : 'SPEED_CALIBRATION_ERROR',
        '13' : 'SPEED_ERROR',
        }

    GetHeatmap(mission['1'],parameters['8'],run,bins,1)

    print("done")

