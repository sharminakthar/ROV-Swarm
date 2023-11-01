from pathlib import Path
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import seaborn as sns


def getHMflock(bins , run):

    path_name0 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/FLOCK_SIZE/9/"+run+"/raw_data_log.csv"
    path_name1 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/FLOCK_SIZE/8/"+run+"/raw_data_log.csv"
    path_name2 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/FLOCK_SIZE/7/"+run+"/raw_data_log.csv"
    path_name3 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/FLOCK_SIZE/6/"+run+"/raw_data_log.csv"
    path_name4 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/FLOCK_SIZE/5/"+run+"/raw_data_log.csv"
    path_name5 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/FLOCK_SIZE/4/"+run+"/raw_data_log.csv"
    path_name6 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/FLOCK_SIZE/3/"+run+"/raw_data_log.csv"


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

    # create figure
    fig, axes = plt.subplots(2, 3)
    axes[0][0].imshow(heatmap0.T, cmap = 'turbo', extent=extent0, origin='lower')
    axes[0][1].imshow(heatmap1.T, cmap = 'turbo', extent=extent1, origin='lower')
    axes[0][2].imshow(heatmap2.T, cmap = 'turbo', extent=extent2, origin='lower')
    axes[1][0].imshow(heatmap3.T, cmap = 'turbo', extent=extent3, origin='lower')
    axes[1][1].imshow(heatmap4.T, cmap = 'turbo', extent=extent4, origin='lower')
    axes[1][2].imshow(heatmap5.T, cmap = 'turbo', extent=extent5, origin='lower')

    axes[0][0].axis('off')
    axes[0][1].axis('off')
    axes[0][2].axis('off')
    axes[1][0].axis('off')
    axes[1][1].axis('off')
    axes[1][2].axis('off')

    axes[0][0].set_title('9 Drones')
    axes[0][1].set_title('8 Drones')
    axes[0][2].set_title('7 Drones')
    axes[1][0].set_title('6 Drones')
    axes[1][1].set_title('5 Drones')
    axes[1][2].set_title('4 Drones')


    plt.show()

def getHMrange_error(bins , run):

    path_name0 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/RANGE_ERROR/0/"+run+"/raw_data_log.csv"
    path_name1 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/RANGE_ERROR/24/"+run+"/raw_data_log.csv"
    path_name2 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/RANGE_ERROR/36/"+run+"/raw_data_log.csv"
    path_name3 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/RANGE_ERROR/48/"+run+"/raw_data_log.csv"
    path_name4 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/RANGE_ERROR/60/"+run+"/raw_data_log.csv"
    path_name5 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/RANGE_ERROR/72/"+run+"/raw_data_log.csv"
    path_name6 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/RANGE_ERROR/84/"+run+"/raw_data_log.csv"
    path_name7 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/RANGE_ERROR/96/"+run+"/raw_data_log.csv"
    path_name8 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/RANGE_ERROR/108/"+run+"/raw_data_log.csv"


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

    data7 = pd.read_csv(path_name7)
    x_p7 = data7["X Position"]
    y_p7 = data7["Y Position"]
    x_array7 = x_p7.to_numpy()
    y_array7 = y_p7.to_numpy()

    data8 = pd.read_csv(path_name8)
    x_p8 = data8["X Position"]
    y_p8 = data8["Y Position"]
    x_array8 = x_p8.to_numpy()
    y_array8 = y_p8.to_numpy()

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

    heatmap7, xedges7, yedges7 = np.histogram2d(x_array7, y_array7, bins=bins)
    extent7 = [xedges7[0], xedges7[-1], yedges7[0], yedges7[-1]]

    heatmap8, xedges8, yedges8 = np.histogram2d(x_array8, y_array8, bins=bins)
    extent8 = [xedges8[0], xedges8[-1], yedges8[0], yedges8[-1]]

    
    # create figure

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "9"


    fig, axes = plt.subplots(3, 3,figsize = [10,9])
    axes[0][0].imshow(heatmap0.T, cmap = 'turbo', extent=extent0, origin='lower')
    axes[0][1].imshow(heatmap1.T, cmap = 'turbo', extent=extent1, origin='lower')
    axes[0][2].imshow(heatmap2.T, cmap = 'turbo', extent=extent2, origin='lower')
    axes[1][0].imshow(heatmap3.T, cmap = 'turbo', extent=extent3, origin='lower')
    axes[1][1].imshow(heatmap4.T, cmap = 'turbo', extent=extent4, origin='lower')
    axes[1][2].imshow(heatmap5.T, cmap = 'turbo', extent=extent5, origin='lower')
    axes[2][0].imshow(heatmap6.T, cmap = 'turbo', extent=extent6, origin='lower')
    axes[2][1].imshow(heatmap7.T, cmap = 'turbo', extent=extent7, origin='lower')
    axes[2][2].imshow(heatmap8.T, cmap = 'turbo', extent=extent8, origin='lower')


    axes[0][0].set_title('0',fontdict={'fontsize': 12})
    axes[0][1].set_title('24',fontdict={'fontsize': 12})
    axes[0][2].set_title('36',fontdict={'fontsize': 12})
    axes[1][0].set_title('48',fontdict={'fontsize': 12})
    axes[1][1].set_title('60',fontdict={'fontsize': 12})
    axes[1][2].set_title('72',fontdict={'fontsize': 12})
    axes[2][0].set_title('84',fontdict={'fontsize': 12})
    axes[2][1].set_title('96',fontdict={'fontsize': 12})
    axes[2][2].set_title('108',fontdict={'fontsize': 12})

    x_locator = FixedLocator([1000, 2000, 3000])
    y_locator = FixedLocator([1000, 2000, 3000, 4000])

    for i in range(3):
        for j in range(3):
            axes[i][j].xaxis.set_major_locator(x_locator)
            axes[i][j].yaxis.set_major_locator(y_locator)


    plt.show()

def getHMbearing_error(bins , run):

    path_name0 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BEARING_ERROR/0/"+run+"/raw_data_log.csv"
    path_name1 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BEARING_ERROR/20/"+run+"/raw_data_log.csv"
    path_name2 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BEARING_ERROR/30/"+run+"/raw_data_log.csv"
    path_name3 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BEARING_ERROR/40/"+run+"/raw_data_log.csv"
    path_name4 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BEARING_ERROR/50/"+run+"/raw_data_log.csv"
    path_name5 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BEARING_ERROR/60/"+run+"/raw_data_log.csv"
    path_name6 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BEARING_ERROR/70/"+run+"/raw_data_log.csv"
    path_name7 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BEARING_ERROR/80/"+run+"/raw_data_log.csv"
    path_name8 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BEARING_ERROR/90/"+run+"/raw_data_log.csv"


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

    data7 = pd.read_csv(path_name7)
    x_p7 = data7["X Position"]
    y_p7 = data7["Y Position"]
    x_array7 = x_p7.to_numpy()
    y_array7 = y_p7.to_numpy()

    data8 = pd.read_csv(path_name8)
    x_p8 = data8["X Position"]
    y_p8 = data8["Y Position"]
    x_array8 = x_p8.to_numpy()
    y_array8 = y_p8.to_numpy()

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

    heatmap7, xedges7, yedges7 = np.histogram2d(x_array7, y_array7, bins=bins)
    extent7 = [xedges7[0], xedges7[-1], yedges7[0], yedges7[-1]]

    heatmap8, xedges8, yedges8 = np.histogram2d(x_array8, y_array8, bins=bins)
    extent8 = [xedges8[0], xedges8[-1], yedges8[0], yedges8[-1]]

    # create figure
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "9"


    fig, axes = plt.subplots(3, 3,figsize = [10,9])
    axes[0][0].imshow(heatmap0.T, cmap = 'turbo', extent=extent0, origin='lower')
    axes[0][1].imshow(heatmap1.T, cmap = 'turbo', extent=extent1, origin='lower')
    axes[0][2].imshow(heatmap2.T, cmap = 'turbo', extent=extent2, origin='lower')
    axes[1][0].imshow(heatmap3.T, cmap = 'turbo', extent=extent3, origin='lower')
    axes[1][1].imshow(heatmap4.T, cmap = 'turbo', extent=extent4, origin='lower')
    axes[1][2].imshow(heatmap5.T, cmap = 'turbo', extent=extent5, origin='lower')
    axes[2][0].imshow(heatmap6.T, cmap = 'turbo', extent=extent6, origin='lower')
    axes[2][1].imshow(heatmap7.T, cmap = 'turbo', extent=extent7, origin='lower')
    axes[2][2].imshow(heatmap8.T, cmap = 'turbo', extent=extent8, origin='lower')


    axes[0][0].set_title('0',fontdict={'fontsize': 12})
    axes[0][1].set_title('20',fontdict={'fontsize': 12})
    axes[0][2].set_title('30',fontdict={'fontsize': 12})
    axes[1][0].set_title('40',fontdict={'fontsize': 12})
    axes[1][1].set_title('50',fontdict={'fontsize': 12})
    axes[1][2].set_title('60',fontdict={'fontsize': 12})
    axes[2][0].set_title('70',fontdict={'fontsize': 12})
    axes[2][1].set_title('80',fontdict={'fontsize': 12})
    axes[2][2].set_title('90',fontdict={'fontsize': 12})

    x_locator = FixedLocator([1000, 2000, 3000])
    y_locator = FixedLocator([1000, 2000, 3000, 4000])

    for i in range(3):
        for j in range(3):
            axes[i][j].xaxis.set_major_locator(x_locator)
            axes[i][j].yaxis.set_major_locator(y_locator)


    plt.show()

def getHMbandwidth(bins , run):

    path_name0 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BANDWIDTH/0.0/"+run+"/raw_data_log.csv"
    path_name1 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BANDWIDTH/0.02/"+run+"/raw_data_log.csv"
    path_name2 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BANDWIDTH/0.03/"+run+"/raw_data_log.csv"
    path_name3 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BANDWIDTH/0.06/"+run+"/raw_data_log.csv"
    path_name4 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BANDWIDTH/0.12/"+run+"/raw_data_log.csv"
    path_name5 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BANDWIDTH/0.25/"+run+"/raw_data_log.csv"
    path_name6 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BANDWIDTH/0.5/"+run+"/raw_data_log.csv"
    path_name7 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BANDWIDTH/1.0/"+run+"/raw_data_log.csv"
    path_name8 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/BANDWIDTH/2.0/"+run+"/raw_data_log.csv"


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

    data7 = pd.read_csv(path_name7)
    x_p7 = data7["X Position"]
    y_p7 = data7["Y Position"]
    x_array7 = x_p7.to_numpy()
    y_array7 = y_p7.to_numpy()

    data8 = pd.read_csv(path_name8)
    x_p8 = data8["X Position"]
    y_p8 = data8["Y Position"]
    x_array8 = x_p8.to_numpy()
    y_array8 = y_p8.to_numpy()

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

    heatmap7, xedges7, yedges7 = np.histogram2d(x_array7, y_array7, bins=bins)
    extent7 = [xedges7[0], xedges7[-1], yedges7[0], yedges7[-1]]

    heatmap8, xedges8, yedges8 = np.histogram2d(x_array8, y_array8, bins=bins)
    extent8 = [xedges8[0], xedges8[-1], yedges8[0], yedges8[-1]]

    # create figure
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "9"


    fig, axes = plt.subplots(3, 3,figsize = [10,9])
    axes[0][0].imshow(heatmap0.T, cmap = 'turbo', extent=extent0, origin='lower')
    axes[0][1].imshow(heatmap1.T, cmap = 'turbo', extent=extent1, origin='lower')
    axes[0][2].imshow(heatmap2.T, cmap = 'turbo', extent=extent2, origin='lower')
    axes[1][0].imshow(heatmap3.T, cmap = 'turbo', extent=extent3, origin='lower')
    axes[1][1].imshow(heatmap4.T, cmap = 'turbo', extent=extent4, origin='lower')
    axes[1][2].imshow(heatmap5.T, cmap = 'turbo', extent=extent5, origin='lower')
    axes[2][0].imshow(heatmap6.T, cmap = 'turbo', extent=extent6, origin='lower')
    axes[2][1].imshow(heatmap7.T, cmap = 'turbo', extent=extent7, origin='lower')
    axes[2][2].imshow(heatmap8.T, cmap = 'turbo', extent=extent8, origin='lower')

    axes[0][0].set_title('0.0',fontdict={'fontsize': 12})
    axes[0][1].set_title('0.02',fontdict={'fontsize': 12})
    axes[0][2].set_title('0.03',fontdict={'fontsize': 12})
    axes[1][0].set_title('0.06',fontdict={'fontsize': 12})
    axes[1][1].set_title('0.12',fontdict={'fontsize': 12})
    axes[1][2].set_title('0.25',fontdict={'fontsize': 12})
    axes[2][0].set_title('0.5',fontdict={'fontsize': 12})
    axes[2][1].set_title('1.0',fontdict={'fontsize': 12})
    axes[2][2].set_title('2.0',fontdict={'fontsize': 12})

    plt.show()

def getHMpacket_loss(bins,run):

    path_name0 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS/70/"+run+"/raw_data_log.csv"
    path_name1 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS/76/"+run+"/raw_data_log.csv"
    path_name2 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS/79/"+run+"/raw_data_log.csv"
    path_name3 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS/82/"+run+"/raw_data_log.csv"
    path_name4 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS/85/"+run+"/raw_data_log.csv"
    path_name5 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS/88/"+run+"/raw_data_log.csv"
    path_name6 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS/91/"+run+"/raw_data_log.csv"
    path_name7 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS/94/"+run+"/raw_data_log.csv"
    path_name8 = "../out/FOLLOW_CIRCLE_ULTRA_EXTENDED_DATA/PACKET_LOSS/97/"+run+"/raw_data_log.csv"


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

    data7 = pd.read_csv(path_name7)
    x_p7 = data7["X Position"]
    y_p7 = data7["Y Position"]
    x_array7 = x_p7.to_numpy()
    y_array7 = y_p7.to_numpy()

    data8 = pd.read_csv(path_name8)
    x_p8 = data8["X Position"]
    y_p8 = data8["Y Position"]
    x_array8 = x_p8.to_numpy()
    y_array8 = y_p8.to_numpy()

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

    heatmap7, xedges7, yedges7 = np.histogram2d(x_array7, y_array7, bins=bins)
    extent7 = [xedges7[0], xedges7[-1], yedges7[0], yedges7[-1]]

    heatmap8, xedges8, yedges8 = np.histogram2d(x_array8, y_array8, bins=bins)
    extent8 = [xedges8[0], xedges8[-1], yedges8[0], yedges8[-1]]

    # create figure
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "9"


    fig, axes = plt.subplots(3, 3,figsize = [10,9])
    axes[0][0].imshow(heatmap0.T, cmap = 'turbo', extent=extent0, origin='lower')
    axes[0][1].imshow(heatmap1.T, cmap = 'turbo', extent=extent1, origin='lower')
    axes[0][2].imshow(heatmap2.T, cmap = 'turbo', extent=extent2, origin='lower')
    axes[1][0].imshow(heatmap3.T, cmap = 'turbo', extent=extent3, origin='lower')
    axes[1][1].imshow(heatmap4.T, cmap = 'turbo', extent=extent4, origin='lower')
    axes[1][2].imshow(heatmap5.T, cmap = 'turbo', extent=extent5, origin='lower')
    axes[2][0].imshow(heatmap6.T, cmap = 'turbo', extent=extent6, origin='lower')
    axes[2][1].imshow(heatmap7.T, cmap = 'turbo', extent=extent7, origin='lower')
    axes[2][2].imshow(heatmap8.T, cmap = 'turbo', extent=extent8, origin='lower')


    axes[0][0].set_title('70',fontdict={'fontsize': 10})
    axes[0][1].set_title('76',fontdict={'fontsize': 10})
    axes[0][2].set_title('79',fontdict={'fontsize': 10})
    axes[1][0].set_title('82',fontdict={'fontsize': 10})
    axes[1][1].set_title('85',fontdict={'fontsize': 10})
    axes[1][2].set_title('88',fontdict={'fontsize': 10})
    axes[2][0].set_title('91',fontdict={'fontsize': 10})
    axes[2][1].set_title('94',fontdict={'fontsize': 10})
    axes[2][2].set_title('97',fontdict={'fontsize': 10})

    plt.show()


if __name__ == "__main__":

    bins = 200
    run = '3'

    #getHMpacket_loss(bins , run)
    getHMbandwidth(bins ,run)
    #getHMbearing_error(bins, run)
    #getHMrange_error(bins , run)
    #getHMflock(bins , run)

    
    print("done")

