from pathlib import Path
import numpy as np
from numpy import sin, cos, pi, linspace
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import os 
import math
from util.vector_helpers import bearing_to_vector


def GetHeatmap(p: Path, mission,para,run, errors = None, save_folder = None):
    bins = 250
    pathname = str(p) +'/'+para
    filenames = os.listdir(pathname)
    newnames = []
    for x in range(len(filenames)):
        print(filenames[x])
        print(errors)
        if  (errors == None) or (filenames[x] in errors) == True :
            newnames.append(filenames[x])
            if newnames[len(newnames)-1].isdigit():
                newnames[len(newnames)-1] = int(newnames[len(newnames)-1])
            else:
                newnames[len(newnames)-1] = float(newnames[len(newnames)-1])
    filenames = sorted(newnames)
    filenames = [str(x) for x in filenames]

    a = math.ceil(len(filenames)/3)

    #fig, axes = plt.subplots(a, 3 ,figsize = [8,8])
    fig = plt.figure()
    nrows = a
    ncols = 3
    for i in range(a):
        if (len(filenames) - i*3) / 3.0 >= 1:
            n = 3 
        else:
            n = (len(filenames) - i*3.0) % 3.0 
        for j in range(int(n)):
      
  

            data0 = pd.read_csv(pathname+'/'+filenames[(i*3)+j]+'/'+str(run)+"/raw_data_log.csv")
            x_p0 = data0["X Position"]


            y_p0 = data0["Y Position"]
            x_array0 = x_p0.to_numpy()
            y_array0 = y_p0.to_numpy()

            heatmap0, xedges0, yedges0 = np.histogram2d(x_array0, y_array0, bins=bins)
            minp = min(xedges0[0],xedges0[-1],yedges0[0],yedges0[-1])
            maxp = max(xedges0[0],xedges0[-1],yedges0[0],yedges0[-1])
            extent0 = [minp,maxp ,minp , maxp]
            x_points = np.linspace(start = minp, stop = maxp, num = 3)


            ax = fig.add_subplot(nrows, ncols, ((i)*3) + (j+1) )


            ax.imshow(heatmap0.T, cmap = 'turbo', extent=extent0, origin='lower')
            ax.set_title(filenames[(i*3)+j],fontdict={'fontsize': 12})
            ax.xaxis.set_major_locator(FixedLocator([round(item, -2) for item in x_points]))

            if mission == 'FOLLOW_CIRCLE':
                axes[i][j].add_patch(Circle((2500,2500),1000, fill = False, color = '#39FF14'))
            elif mission == 'RACETRACK':
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

                ax.plot(arc_xs, arc_ys, color = '#39FF14', lw = lw)
                ax.plot(arc_xs2, arc_ys2, color = '#39FF14', lw = lw)
                ax.plot( x , y , lw=lw , color = '#39FF14')
                ax.plot( x2 , y2 , lw=lw ,color = '#39FF14' )

            elif mission == "FIXED_HEADING":

                pos = bearing_to_vector(325).reshape((2, 1))

                xmin = np.amin(x_array0)
                xmax = np.amax(x_array0)
                print(xmin)
                print(xmax)
                x = np.linspace(xmin,xmax,100)
                y = pos[0]/pos[1] * x   
                ax.plot(x, y, color = '#39FF14', lw = 1)



    fig.suptitle('Heatmap for ' + mission +  ' under varying '+str.title(para.replace("_", " " )), fontsize=16)
    plt.tight_layout()

    if save_folder == None:
        save_folder = "Graphs"

    folder = p /  save_folder /  "Heatmaps"
    
    folder.mkdir(parents=True, exist_ok=True)


    fig.savefig(( folder /  (para + "_" +  str(run) +".png")), bbox_inches="tight")




def GetSingleHeatmap(p: Path, mission,para,value,run,bins):
    pathname = str(p) + '/'+para+'/'+value+'/'+str(run)+'/'+"/raw_data_log.csv"

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




