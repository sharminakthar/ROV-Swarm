from pathlib import Path
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    bins = 200

    run = '2'

    path_name0 = "../FOLLOW_CIRCLE/FLOCK_SIZE/9/"+run+"/raw_data_log.csv"
    path_name1 = "../FOLLOW_CIRCLE/FLOCK_SIZE/8/"+run+"/raw_data_log.csv"
    path_name2 = "../FOLLOW_CIRCLE/FLOCK_SIZE/7/"+run+"/raw_data_log.csv"
    path_name3 = "../FOLLOW_CIRCLE/FLOCK_SIZE/6/"+run+"/raw_data_log.csv"
    path_name4 = "../FOLLOW_CIRCLE/FLOCK_SIZE/5/"+run+"/raw_data_log.csv"
    path_name5 = "../FOLLOW_CIRCLE/FLOCK_SIZE/4/"+run+"/raw_data_log.csv"
    path_name6 = "../FOLLOW_CIRCLE/FLOCK_SIZE/3/"+run+"/raw_data_log.csv"


    data0 = pd.read_csv(path_name0)
    x_p0 = data0["X Position"]
    y_p0 = data0["Y Position"]
    x_array0 = x_p0.to_numpy()
    y_array0 = y_p0.to_numpy()

    data1 = pd.read_csv(path_name1)
    x_p1 = data1["X Position"]
    y_p1 = data1["Y Position"]
    x_array1 = x_p1.to_numpy()
    y_array1 = y_p1.to_numpy()

    data2 = pd.read_csv(path_name2)
    x_p2 = data2["X Position"]
    y_p2 = data2["Y Position"]
    x_array2 = x_p2.to_numpy()
    y_array2 = y_p2.to_numpy()

    data3 = pd.read_csv(path_name3)
    x_p3 = data3["X Position"]
    y_p3 = data3["Y Position"]
    x_array3 = x_p3.to_numpy()
    y_array3 = y_p3.to_numpy()

    data4 = pd.read_csv(path_name4)
    x_p4 = data4["X Position"]
    y_p4 = data4["Y Position"]
    x_array4 = x_p4.to_numpy()
    y_array4 = y_p4.to_numpy()

    data5 = pd.read_csv(path_name5)
    x_p5 = data5["X Position"]
    y_p5 = data5["Y Position"]
    x_array5 = x_p5.to_numpy()
    y_array5 = y_p5.to_numpy()

    data6 = pd.read_csv(path_name6)
    x_p6 = data6["X Position"]
    y_p6 = data6["Y Position"]
    x_array6 = x_p6.to_numpy()
    y_array6 = y_p6.to_numpy()

    heatmap0, xedges0, yedges0 = np.histogram2d(x_array0, y_array0, bins=bins)
    extent0 = [xedges0[0], xedges0[-1], yedges0[0], yedges0[-1]]

    heatmap1, xedges1, yedges1 = np.histogram2d(x_array1, y_array1, bins=bins)
    extent1 = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]

    heatmap2, xedges2, yedges2 = np.histogram2d(x_array2, y_array2, bins=bins)
    extent2 = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]

    heatmap3, xedges3, yedges3 = np.histogram2d(x_array3, y_array3, bins=bins)
    extent3 = [xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]]

    heatmap4, xedges4, yedges4 = np.histogram2d(x_array4, y_array4, bins=bins)
    extent4 = [xedges4[0], xedges4[-1], yedges4[0], yedges4[-1]]

    heatmap5, xedges5, yedges5 = np.histogram2d(x_array5, y_array5, bins=bins)
    extent5 = [xedges5[0], xedges5[-1], yedges5[0], yedges5[-1]]

    heatmap6, xedges6, yedges6 = np.histogram2d(x_array6, y_array6, bins=bins)
    extent6 = [xedges6[0], xedges6[-1], yedges6[0], yedges6[-1]]


    # plt.clf()
    # plt.imshow(heatmap.T, cmap = 'turbo', extent=extent, origin='lower')
    # plt.show()

    # create figure
    fig = plt.figure(figsize=(10, 7))
  
    # setting values to rows and column variables
    rows = 2
    columns = 4

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
  
    # showing image
    plt.imshow(heatmap0.T, cmap = 'turbo', extent=extent0, origin='lower')
    plt.axis('off')
    plt.title("9 Drones")
  
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
  
    # showing image
    plt.imshow(heatmap1.T, cmap = 'turbo', extent=extent1, origin='lower')
    plt.axis('off')
    plt.title("8 Drones")
  
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
  
    # showing image
    plt.imshow(heatmap2.T, cmap = 'turbo', extent=extent2, origin='lower')
    plt.axis('off')
    plt.title("7 Drones")
  
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)
  
    # showing image
    plt.imshow(heatmap3.T, cmap = 'turbo', extent=extent3, origin='lower')
    plt.axis('off')
    plt.title("6 Drones")

    # Adds a subplot at the 5nd position
    fig.add_subplot(rows, columns, 5)
  
    # showing image
    plt.imshow(heatmap4.T, cmap = 'turbo', extent=extent4, origin='lower')
    plt.axis('off')
    plt.title("5 Drones")
  
    # Adds a subplot at the 6rd position
    fig.add_subplot(rows, columns, 6)
  
    # showing image
    plt.imshow(heatmap5.T, cmap = 'turbo', extent=extent5, origin='lower')
    plt.axis('off')
    plt.title("4 Drones")
  
    # Adds a subplot at the 7th position
    fig.add_subplot(rows, columns, 7)
  
    # showing image
    plt.imshow(heatmap6.T, cmap = 'turbo', extent=extent6, origin='lower')
    plt.axis('off')
    plt.title("3 Drones")

    plt.show()


    print("done")

