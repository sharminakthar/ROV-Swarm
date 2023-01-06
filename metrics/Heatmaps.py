from pathlib import Path
import numpy as np
from numpy import sin, cos, pi, linspace
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
    a = 3
    filenames = ['0','10','20','30','40','50','60','70','90']

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

            if path and mission == 'FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA':
                axes[i][j].add_patch(Circle((2500,2500),1000, fill = False, color = '#39FF14'))
            elif path and mission == 'RACETRACK_EXTENDED':
                r = 1000
                lw = 1

                arc_angles = linspace(pi/2, 2*pi*(3/4), 30)
                arc_xs = r * cos(arc_angles) + 1500
                arc_ys = r * sin(arc_angles) + 2500

                arc_angles2 = linspace(-1*pi*(1/2),pi/2, 30)
                arc_xs2 = r * cos(arc_angles2) + 3500
                arc_ys2 = r * sin(arc_angles2) + 2500

                p1 = [1500,1500]
                p2 = [3500,1500]
                x, y = [p1[0], p2[0]], [p1[1], p2[1]]

                p3 = [1500,3500]
                p4 = [3500,3500]
                x2, y2 = [p3[0], p4[0]], [p3[1], p4[1]]

                axes[i][j].plot(arc_xs, arc_ys, color = '#39FF14', lw = lw)
                axes[i][j].plot(arc_xs2, arc_ys2, color = '#39FF14', lw = lw)
                axes[i][j].plot( x , y , lw=lw , color = '#39FF14')
                axes[i][j].plot( x2 , y2 , lw=lw ,color = '#39FF14' )

    fig.suptitle('Heatmap for Follow Circle under varying '+str.title(para.replace("_", " " )), fontsize=16)
    plt.tight_layout()
    fig.savefig((para + run +".png"), bbox_inches="tight")


def GetSingleHeatmap(mission,para,value,run,bins,path):
    pathname = "../out/"+mission+'/'+para+'/'+value+'/'+run+'/'+"/raw_data_log.csv"

    data0 = pd.read_csv(pathname)
    x_p0 = data0["X Position"]
    y_p0 = data0["Y Position"]
    x_array0 = x_p0.to_numpy()
    y_array0 = y_p0.to_numpy()

    heatmap0, xedges0, yedges0 = np.histogram2d(x_array0, y_array0, bins=bins)
    minp = min(xedges0[0],xedges0[-1],yedges0[0],yedges0[-1])
    maxp = max(xedges0[0],xedges0[-1],yedges0[0],yedges0[-1])
    extent0 = [minp,maxp ,minp , maxp]
    x_points = np.linspace(start = minp, stop = maxp, num = 3)

    plt.imshow(heatmap0.T, cmap = 'turbo', extent=extent0, origin='lower')
    #plt.xaxis.set_major_locator(FixedLocator([round(item, -2) for item in x_points]))

    if path and mission == 'FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA':
        plt.add_patch(Circle((2500,2500),1000, fill = False, color = '#39FF14'))
    elif path and mission == 'RACETRACK_EXTENDED':
        r = 1000
        lw = 1

        arc_angles = linspace(pi/2, 2*pi*(3/4), 30)
        arc_xs = r * cos(arc_angles) + 1500
        arc_ys = r * sin(arc_angles) + 2500

        arc_angles2 = linspace(-1*pi*(1/2),pi/2, 30)
        arc_xs2 = r * cos(arc_angles2) + 3500
        arc_ys2 = r * sin(arc_angles2) + 2500

        p1 = [1500,1500]
        p2 = [3500,1500]
        x, y = [p1[0], p2[0]], [p1[1], p2[1]]

        p3 = [1500,3500]
        p4 = [3500,3500]
        x2, y2 = [p3[0], p4[0]], [p3[1], p4[1]]

        plt.plot(arc_xs, arc_ys, color = '#39FF14', lw = lw)
        plt.plot(arc_xs2, arc_ys2, color = '#39FF14', lw = lw)
        plt.plot( x , y , lw=lw , color = '#39FF14')
        plt.plot( x2 , y2 , lw=lw ,color = '#39FF14' )

        plt.show()




if __name__ == "__main__":

    bins = 250
    run = '4'
    mission = {
        '1' : 'FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA',
        '2' : 'FIXED_HEADING',
        '3' : 'RACETRACK_EXTENDED'
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

    GetSingleHeatmap(mission['3'],parameters['3'],'1.0','0',bins,1)



    print("done")

