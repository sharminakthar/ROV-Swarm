from pathlib import Path
import pandas as pd
import numpy as np
from metrics.BaseMetric import BaseMetric
from metrics.CentreDistMetric import CentreDistMetric
from metrics.Separation import Separation

from metrics.collisionsnumber import CollisionsNumber
from metrics.Density import Density
from metrics.orientations import OrientationMetric
from metrics.PerceivedPosMetric import PerceivedPosMetric
from metrics.speed import Speed
from metrics.Fixed_Heading_traj import FHTrajectoryMetric
#from metrics.trajectories import TrajectoryMetric
import seaborn as sb
from scipy.stats import spearmanr

from metrics.Helper_Functions import moving_average
from metrics.trajectories import TPTrajectoryMetric
#from metrics.trajectories import FHTrajectoryMetric
import numpy as np
from textwrap import wrap
import seaborn as sb
from matplotlib import cm
import os
import sys
import shutil
from metrics.DistfromRT import distfromRT
from matplotlib import pyplot as plt
import os
#impo
#from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
import itertools



units_list = {
    "ACCELERATION_CALIBRATION_ERROR": "m/s$^2$",
    "ACCELERATION_ERROR": "m/s$^2$",
    "BANDWIDTH": "B/s",
    "FLOCK_SIZE": "",
    "BEARING_CALIBRATION_ERROR": "$^\circ$",
    "BEARING_ERROR": "$^\circ$",
    "HEADING_CALIBRATION_ERROR": "$^\circ$",
    "HEADING_ERROR": "$^\circ$",
    "PACKET_LOSS": "%",
    "RANGE_CALIBRATION_ERROR": "m",
    "RANGE_ERROR": "m",
    "SPEED_CALIBRATION_ERROR": "m/s",
    "SPEED_ERROR": "m/s"
}

class Grapher():
    
    def __init__(self):
        pass

    def get_single_val_data(self, metric: BaseMetric, directory: Path) -> pd.DataFrame:
        """
        Calculates the values of the metric on at a single value.

        Parameters
            metric_list: BaseMetric
                The metric to be run on the data.
            directory: Path
                The directory of the data being read.
        
        Returns
            The dataframe linking the number of the run to the dataframe containing the values of the metric
            at each timestep for each run.
        """
        results = {}
        dirs = filter(lambda x: x.is_dir(), directory.iterdir())
        reduction = "mean"
        for i, run in enumerate(dirs):
            run = run / "raw_data_log.csv"
            data = pd.read_csv(run)
            result = metric.calculate(data)
            if i == 0:
                results["Timestep"] = result.iloc[:, 0]
            results[str(i)] = result.iloc[:, 1]
    
        return pd.DataFrame(results)

    def get_single_var_data(self, metric: BaseMetric, directory: Path, reduction="mean") -> dict[str, pd.DataFrame]:
        """
        Takes in a list of metrics and a directory of a single variable and produces the metric data 
        of the simulations in that directory. There will be the data for a single graph for each metric.

        Parameters
            metric_list: BaseMetric
                The metric to be run on the data.
            directory: Path
                The directory of the data being read.
            reduction: str
                The method used to reduce the data of the runs into one list of values. Should either be "mean"
                or "runN" where N is the number of the run to extract.
            
        Returns
            A dictionary linking each variable value to the values of the metric at each timestep.
        """
        ind_var_output = {}
        reduction = "mean"
        for ind_var in directory.iterdir():
            df = self.get_single_val_data(metric, ind_var)
            result = None
            if reduction == "mean":
                result = pd.concat([df["Timestep"], df.loc[:, df.columns != "Timestep"].mean(axis=1)], axis=1)
            elif reduction.startswith("run"):
                result = pd.concat([df["Timestep"], df[reduction[3:]]], axis=1)
            elif reduction == "none":
                result = df
            else:
                raise KeyError("Expected one of 'mean' or 'runN' for the data reduction")
            ind_var_output[ind_var.name] = result
        
        return ind_var_output


    def get_all_var_data(self, metric_list: dict, directory: Path, reduction: str="mean", graph_func=None,
                         save_data: bool=True, specific_param: list=None, justGraphs: bool = False, save_folder: str=None, **kwargs):
        """
        Takes in a list of metrics and a directory of multilpe variables changing and produces the metric data 
        of the simulations in that directory. There will be data for a set of graphs for each variable for each metric.

        Parameters
            metric_list: dict
                A dictionary linking the name of a metric to a metric information.
            directory: Path
                The directory of the data being read.
            reduction: str
                The method used to reduce the data of the runs into one list of values. Should either be "mean"
                or "runN" where N is the number of the run to extract.
            graph_func:
                The function used to produce a graph from the data
            save_data: bool
                A flag that will denote whether or not to save the raw data from the metrics.
            specific_param: list
                A list of specific parameters to run (e.g. ["BANDWIDTH", "PACKET_LOSS"]) if only some of the parameters
                are needed to be run. If None, then all of the parameters will be run
            save_folder:
                If you would rather save the graph and metric data in subfolders so that they are easily accessible, 
                name that folder here. If None, will save directly to Graphs/var/metric/
            **kwargs:
                Keyword arguments for the graph function
        """
        data_folder = directory / "Metric_Data"
        fig_folder = directory / "Graphs"
        if save_data:
            
            data_folder.mkdir(parents=True, exist_ok=True)
        if graph_func:
            fig_folder.mkdir(parents=True, exist_ok=True)
        

        print("BEGIN")
        for var in directory.iterdir():
            if var.name in ["Graphs", "Metric_Data"]:
                
                continue
            

            print("YES")
            if specific_param is not None and not var.name in specific_param:
                print(var.name)
                continue
          
            for metric_name, metric_info in metric_list.items():
                if justGraphs == False:


                    metric = metric_info["instance"]
                    name = "{} - {}".format(var.name, metric_name)
                    print(name)
                    if metric_name == "col_num":
                        reduction="none"

                    #fdr = data_folder / var.name 
                    if  os.path.exists(data_folder / var.name / metric_name) == False:

                    
                        ind_var_output = self.get_single_var_data(metric, var, reduction=reduction)
                        if save_data:
                            folder = data_folder / var.name / metric_name    
                    

                            for k, v in ind_var_output.items():
                                f = folder / k
                                if os.path.exists(f/"metric_data.csv") == False:
                                    print(f)
                                    f.mkdir(parents=True, exist_ok=True)
                                    v.to_csv(path_or_buf=(f / "metric_data.csv"), index=False)

                        if graph_func is not None:
                            if metric_name == "col_num":
                                fig = graph_func(self, ind_var_output, metric_info, var.name,  "sum", **kwargs)
                            else:
                                fig = graph_func(self, ind_var_output, metric_info, var.name, "mean", **kwargs)
                            folder = fig_folder / var.name / metric_name
                            if save_folder is not None:
                                folder = folder / save_folder
                            folder.mkdir(parents=True, exist_ok=True)
                            fig.savefig(folder / (metric_name + ".png"), bbox_inches="tight")
                            plt.close(fig)
                else:
                    ind_var_output = {}
                    folder = data_folder / var.name / metric_name 
                    if graph_func is not None:
                        for k in folder.iterdir():
                            #if float(k.name) < 51:
                                f =  k
                                df = pd.read_csv(f/ "metric_data.csv")
                                ind_var_output[k.name] = df
                        
                        print("WEE")
                        if metric_name == "col_num":
                            fig = graph_func(self, ind_var_output, metric_info, var.name,  "sum", **kwargs)
                        else:
                            fig = graph_func(self, ind_var_output, metric_info, var.name, "mean", **kwargs)

                        folder = fig_folder / var.name / metric_name
                        if save_folder is not None:
                            folder = folder / save_folder
                        folder.mkdir(parents=True, exist_ok=True)
                        fig.savefig(folder / (metric_name + ".png"), bbox_inches="tight")   
                        plt.close(fig)




    
    def generate_smooth_line_chart(self, data, metric_info, var, ignore):

        var_name = " ".join([w.capitalize() for w in var.split("_")])
        fig = plt.figure()
        ax = fig.add_subplot()
        labels = sorted(list(data.keys()), key=lambda x: float(x))
        for k in labels:

            #second parameter is the number adjacent values each data point is averaged with
            k2 = moving_average(data[k], 3)

            d = k2
            ax.plot(d["Timestep"].to_numpy(), d.loc[:, d.columns != "Timestep"].to_numpy(), label=k)
        ax.set_title("{} with varying {}".format(metric_info["desc"], var_name), wrap=True)
        ax.set_xlabel("Timesteps (s)")
        ax.set_ylabel("{} ({})".format(metric_info["axis_label"], metric_info["unit"]))
        legend_title = "{} ({})".format(var_name, units_list[var])
        legend_title = "\n".join(wrap(legend_title, 20))
        l = ax.legend(title=legend_title, bbox_to_anchor=(1.04,1), loc="upper left")
        plt.setp(l.get_title(), multialignment='center')
        return fig


    def generate_line_chart(self, data, metric_info, var, ignore):
        var_name = " ".join([w.capitalize() for w in var.split("_")])

        fig = plt.figure()
        ax = fig.add_subplot()

        labels = sorted(list(data.keys()), key=lambda x: float(x))
        for k in labels:
            d = data[k]
            ax.plot(d["Timestep"].to_numpy(), d.loc[:, d.columns != "Timestep"].to_numpy(), label=k)

        ax.set_title("{} with varying {}".format(metric_info["desc"], var_name), wrap=True)
        ax.set_xlabel("Timesteps (s)")
        ax.set_ylabel("{} ({})".format(metric_info["axis_label"], metric_info["unit"]))

        legend_title = "{} ({})".format(var_name, units_list[var])
        legend_title = "\n".join(wrap(legend_title, 20))
        l = ax.legend(title=legend_title, bbox_to_anchor=(1.04,1), loc="upper left")
        plt.setp(l.get_title(), multialignment='center')

        return fig

    

    def generate_bar_chart(self, data, metric_info, var, bar_reduction):
        """
        bar_reduction is one of "mean", "sum", "last", "lastN"
        "lastN" takes the mean of the last N timesteps, where N can be any number 0 < N < 10000
        """
       # bar_reduction = "std"
        func = determine_bar_reduction(bar_reduction)

        var_name = " ".join([w.capitalize() for w in var.split("_")])
        fig = plt.figure()
        ax = fig.add_subplot()
        labels = sorted(list(data.keys()), key=lambda x: float(x))
        heights = []
        for k in labels:
            d = data[k]
            heights.append(func(d.loc[:, d.columns != "Timestep"].to_numpy()))

        #labels = labels[:len(labels)-4]
        #heights = heights[:len(heights)-4]
        ax.bar(labels, heights)
        ax.set_title("{} with varying {}".format(metric_info["desc"], var_name), wrap=True)
        ax.set_xlabel("{} ({})".format(var_name, units_list[var]))
        ax.set_ylabel("{} ({})".format(metric_info["axis_label"], metric_info["unit"]))

        return fig
    
def read_multi_data( p: Path, metric_name="cdm", reduction="mean"):
        reduction = determine_bar_reduction(reduction)
        names = p.name.split("-")[1:]
        data_path = p / "Metric_data"
        axis1 = [n.name for n in data_path.iterdir()]

        for n in data_path.iterdir():
            if len(n.name) < 2:
                if os.path.exists(str(data_path) + "\\" +  "0" + (n.name)):
                    shutil.rmtree(str(data_path) + "\\" +  n.name)
                else:
                    os.rename(str(data_path) + "\\" +  n.name, str(data_path) + "\\" +  "0" + (n.name))


        
        
        axis1_index = {n:i for i,n in enumerate(axis1)}
        axis2 = []
        for n in next(next(data_path.iterdir()).iterdir()).iterdir():
           
            if (n.name  in ["00", "70", "80", "90", "60"]) == False:
                axis2.append(n.name)
        axis2_index = {n:i for i,n in enumerate(axis2)}
        
        data_arr = np.zeros((len(axis1), len(axis2)))
        for packet_loss in data_path.iterdir():

            

            for metric in packet_loss.iterdir():
                for n in metric.iterdir():
                    if len(n.name) < 2:
                        if os.path.exists(str(metric) + "\\" +  "0" + (n.name)):
                            shutil.rmtree(str(metric) + "\\" + n.name)
                        else:
                            os.rename(str(metric) + "\\" +  n.name, str(metric) + "\\" +  "0" + (n.name))
          
                if metric.name == metric_name:
                  
                    for bandwidth in metric.iterdir():
                        
                        if (bandwidth.name in ["90", "80", "70", "00", "60"]) == False :
                            data = pd.read_csv(bandwidth / "metric_data.csv")
                            data_arr[axis1_index[(packet_loss.name)], axis2_index[(bandwidth.name)]] = reduction(data.iloc[:,1].to_numpy())
        return axis1, axis2, data_arr  


def multivar_grapher( metric_list: dict, directory: Path, reduction: str="mean", graph_func1=None, graph_func2=None, graph_func3 = None):
        fig_folder = directory / "Graphs"  
        if graph_func1:
            fig_folder.mkdir(parents=True, exist_ok=True)
        
    
        for metric_name, metric_info in metric_list.items():
                metric = metric_info["instance"]
                print("metric name: ",metric_name)
                if metric_name =="col_num":
                    axis1, axis2, arr = read_multi_data(directory, metric_name, reduction="sum")
                else:
                    axis1, axis2, arr = read_multi_data(directory, metric_name)
                
                if graph_func1 is not None: 
                    fig = graph_func1(arr.astype(float), np.array(axis1).astype(float), np.array(axis2).astype(float))
                    folder = fig_folder / metric_name
                    folder.mkdir(parents=True, exist_ok=True)
                    fig.savefig(folder / (metric_name + "_BAR" +  ".png"), bbox_inches="tight")
                    plt.close(fig)
                if graph_func2 is not None:
                    fig = graph_func2(arr.astype(float), np.array(axis1).astype(float), np.array(axis2).astype(float))
                    folder = fig_folder / metric_name 
                    folder.mkdir(parents=True, exist_ok=True)
                    fig.savefig(folder / (metric_name + "_MAP" +  ".png"), bbox_inches="tight")
                    plt.close(fig)
                if graph_func3 is not None: 
                    fig = graph_func3(arr.astype(float), np.array(axis1).astype(float), np.array(axis2).astype(float))
                    folder = fig_folder / metric_name
                    folder.mkdir(parents=True, exist_ok=True)
                    fig.savefig(folder / (metric_name + "_Surface" +  ".png"), bbox_inches="tight")
                    plt.close(fig)



def determine_bar_reduction(bar_reduction):
    func_dict = {
        "mean": np.mean,
        "sum": np.sum,
        "last": lambda x: x[-1,0],
        "std" : np.std
    }
    if bar_reduction in func_dict.keys():
        func = func_dict[bar_reduction]
    elif bar_reduction.startswith("last"):
        try:
            num = int(bar_reduction[4:])
        except:
            raise Exception("The 'N' of lastN must be a number!")
        assert 0 < num < 10000, "'N' in lastN must be between 0 and 10000!"
        func = lambda x: np.mean(x[-num:-1]) 
    else:
        raise Exception("reduction must be one of 'mean', 'sum', 'last', or 'lastN'")
    
    return func

def generate_MultiVar_heatmap( dataArray, xlabels, ylabels):
        #var1_name = " ".join([w.capitalize() for w in var1.split("_")])
        #var2_name = " ".join([w.capitalize() for w in var2.split("_")])

        #mvData = pd.DataFrame(dataArray, columns=xlabels, index=ylabels).pivot(("{} ({})".format(var1_name, units_list[var1])), ("{} ({})".format(var2_name, units_list[var2])), "{} ({})".format(metric_info["axis_label"], metric_info["unit"]))
        #ylabels = [1.0, 0.88, 0.77, 0.65, 0.53,0.41,0.3,0.18, 0.08, 0.03]
        #ylabels = [0.88,0.77, 0.65, 0.53,0.41]
        xlabels = np.round_(xlabels, decimals = 2)
        ylabels = np.round_(ylabels, decimals = 2)
    
        mvData = pd.DataFrame(dataArray, columns=ylabels, index=xlabels)



        minValue = mvData.min().min()
        maxValue = mvData.max().max()
        fig = sb.heatmap(mvData, vmin = minValue, vmax = maxValue).get_figure()
        #plt.show()
        return fig

def generate_3DBarChart( dataArray, ylabels, xlabels):
        #var1_name = " ".join([w.capitalize() for w in var1.split("_")])
        #var2_name = " ".join([w.capitalize() for w in var2.split("_")])

        #xlabels = [0.88,0.77, 0.65, 0.53,0.41]

        fig = plt.figure(figsize=(8, 3))
        xlabels = np.round_(xlabels, decimals = 2)
        ylabels = np.round_(ylabels, decimals = 2)
        ax1 = fig.add_subplot(121, projection='3d')
        _x = xlabels
        _y = ylabels
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()        
        top = dataArray.ravel()
        
        bottom = np.zeros_like(top)        
        width = xlabels[len(xlabels)-1]/len(xlabels)
        #width = 1/1
        depth = ylabels[len(ylabels)-1]/len(ylabels)
        ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
        #ax1.set_title("{} with varying {}".format(metric_info["desc"], var_name), wrap=True)
        #ax1.set_xlabel("{} ({})".format(var1_name, units_list[var1]))
        #ax1.set_ylabel("{} ({})".format(var2_name, units_list[var2]))
        #ax1.set_zlabel("{} ({})".format(metric_info["axis_label"], metric_info["unit"]))

        #plt.show()

        return fig


def generate_3D_Contour_Plot(dataArray, ylabels, xlabels):
    print(xlabels)
    #xlabels = [1.0, 0.88, 0.77, 0.65, 0.53,0.41,0.3,0.18,0.03]
    #xlabels = [0.88, 0.77, 0.65, 0.53,0.41]
    xlabels = np.round_(xlabels, decimals = 2)
    ylabels = np.round_(ylabels, decimals = 2)
    _x = xlabels
    _y = ylabels
    x, y = np.meshgrid(_x, _y)
       
    top = dataArray
    
    
    fig = plt.figure()
    ax = plt.axes(projection = "3d")
 
    #ax.plot_surface(x, y, top, cmap = cm.coolwarm, lw=0.5, rstride=1, cstride=1)
    ax.contour3D(x, y, top, cmap="binary")

    ax.plot_surface(x, y, top, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
    ax.contour(x, y, top, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
    #ax.contour(x, y,top, 10, lw=3, colors="k", linestyles="solid")
    #ax.contour3D(X, Y, Z, cmap="binary")
    return fig

def getCorrelations(metric_list: dict, directory: Path):
    data_path = directory/"Metric_Data"

    numOfMetrics = len(metric_list)
    allCoeffs = np.empty([0,numOfMetrics])
    vars = []

    for var in data_path.iterdir():
        vars.append(var.name)
        coeffs = []
        metrics = []

        for metric in var.iterdir():    
            metrics.append(metric.name)
            vals = []
            labels = []

            for val in metric.iterdir():
                data = pd.read_csv(val / "metric_data.csv")
                l = data.iloc[:,1].to_numpy()
                if metric.name == "col_num":
                    meanVal = np.sum(data.iloc[:,1].to_numpy())
                else:
                    meanVal = np.mean(data.iloc[:,1].to_numpy())
                vals.append(meanVal)
                labels.append(val.name)
                
            if metric.name=="col_num":
                print("VAR: ", var.name)
                print(determineCoefficient(labels, vals))
            coeffs.append(determineCoefficient(labels, vals))

        print(var.name)
        allCoeffs = np.vstack([allCoeffs, coeffs])

    print(allCoeffs)
    allCoeffs = zip(*allCoeffs)
    print(allCoeffs)

    i = 0

    for metricCoeffs in allCoeffs:
        print("METRIC: ", metrics[i] )
        i += 1
        sortedVars = sort_list(vars, metricCoeffs)
        metricCoeffs = np.sort(metricCoeffs)
        j = len(vars) - 1
        for var in sortedVars:
            print(sortedVars[j], ": ", metricCoeffs[j])
            j-= 1




def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs)]
 
    return z 


def determineCoefficient(d1, d2):
    corr, _ = spearmanr(d1,d2)
    return corr


def regressionGrad(X, y):
    #x - xlabels and y labels
    #y = distacne from rt
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    predicted = regr.predict([[90, 1]])
    arr = []
    for x in range(10):
        predicted = regr.predict([[x*10, 2.2 - (x/10) * 2]])
        arr.append(1 - predicted)
    return arr



if __name__ == "__main__":
    grapher = Grapher()
    metric_list = {                  
                    "sep_min": {
                        "desc": "Minimum separation between drones",
                        "unit": "m",
                        "axis_label": "Minimum Drone Separation",
                        "instance": Separation()
                        },
                   "sep_max": {
                        "desc": "Maximum separation between drones",
                        "unit": "m",
                        "axis_label": "Maximum Drone Separation",
                        "instance": Separation(reduction="max")
                        },
                   "sep_mean": {
                        "desc": "Mean separation between drones",
                        "unit": "m",
                        "axis_label": "Mean Drone Separation",
                        "instance": Separation(reduction="mean")
                       },
                   "col_num": {
                        "desc": "Total number of collisions",
                         "unit": "",
                        "axis_label": "Number of Collisions",
                        "instance": CollisionsNumber()
                       },
                    #"density": {
                    #     "desc": "Density of the swarm",
                    #     "unit": "m$^2$",
                    #     "axis_label": "Swarm Density",
                    #     "instance": Density()
                    #    },
                    "orient": {
                        "desc": "S.D of drone orientations",
                        "unit": "$^\circ$",
                        "axis_label": "Drone Orientation S.D",
                        "instance": OrientationMetric()
                        },
                    "pos_err": {
                        "desc": "Calculated position error",
                        "unit": "m",
                        "axis_label": "Calculated Position Error",
                        "instance": PerceivedPosMetric()
                        },
                    "speed": {
                       "desc": "Speed of drones",
                        "unit": "m/s",
                        "axis_label": "Speed",
                       "instance": Speed()
                        },
                   ##  "dfc": {
                   #      "desc": "Distance from flock center",
                   #      "unit": "m",
                   #      "axis_label": "Distance from flock center",
                   #      "instance": distfromRT()
                   #     },     
                    #"TPtraj": {
                    #     "desc": "Difference from optimal trajectory",
                    #     "unit": "$^\circ$",
                    #     "axis_label": "Angle From Optimal Trajectory",
                    #     "instance": TPTrajectoryMetric()
                    #    },
                    "distfromRT": {
                         "desc": "Distance from Racetrack",
                         "unit": "m",
                         "axis_label": "Distance from Racetrack",
                         "instance": distfromRT()
                        },
                    #"distfromCircle": {
                    #     "desc": "Distance from Circle",
                    #     "unit": "m",
                    #     "axis_label": "Distance from Circle",
                    #     "instance": circleCentreDist()
                    #    },
                    #"FHtraj" : {
                    #    "desc": "Difference from optimal trajectory",
                    #    "unit": "$^\circ$",
                    #    "axis_label": "Angle from Optimal trajectory",
                    #    "instance": FHTrajectoryMetric()
                    #}

                    
    }



    #print("start")
    # Path to the data that is 
    # being graphed
    p = Path("out/newrt")
  #  p = Path("out/RTSING2")
    # Will write all metric data and make graphs automatically
    # Graph_func should have the same parameters as the defined ones, and as many keyword arguments
    # (e.g. "bar_reduction") as needed
    # Example for making a bar chart with the mean of the last 100 values in the run:
    #grapher.get_all_var_data(metric_list, p, graph_func=grapher.generate_bar_chart,
    #                          bar_reduction="sum", save_folder="bar", save_data = True)
    #grapher.get_all_var_data(metric_list, p, graph_func=grapher.generate_smooth_line_chart,
#                            save_folder = "line", save_data = True)


    #grapher.get_all_var_data(metric_list, p)

    # Example of running a line graph on all of the data:
    #grapher.get_all_var_data(metric_list, p, graph_func=grapher.generate_line_chart, save_folder="bar")
    #grapher.get_all_var_data(metric_list, p)

    grapher.get_all_var_data(metric_list, p, save_folder="line", justGraphs=False, graph_func=Grapher.generate_line_chart)
    #grapher.get_all_var_data(metric_list, p, save_folder="bar", justGraphs=False, graph_func=Grapher.generate_bar_chart)


    #axis1, axis2, arr = grapher.read_multi_data(Path("out/FIXED_RACETRACK_MULTIVAR/FLOCK_SIZE-PACKET_LOSS-BANDWIDTH/5"), "cdm")
    #print(axis1)
    #print(axis2)
    #print(arr)
    #metric_name, metric_info = metric_list
    #for metric_name, metric_info in metric_list.items():
     #       metric = metric_info["instance"]
      #      name = "{} - {}".format(var.name, metric_name)
       #     print(name)

    #for metric_name, metric_info in metric_list.items():
    #multivar_grapher(metric_list, p, graph_func1 = generate_3DBarChart, graph_func2 = generate_MultiVar_heatmap, graph_func3 = generate_3D_Contour_Plot)
    '''
    metric_name = "distfromRT"
    axis1, axis2, arr = read_multi_data(p, metric_name) 
    print(axis1)
    c = list(itertools.product(axis1, axis2))
    output = arr.flatten()s
        #print(output)
    c1 = list(zip(*c))[0]
    c2 = list(zip(*c))[1]

    ##print(c1)
    #print(c2)
    df = pd.DataFrame({'x':c1, 'y':c2})

    df["output"] = output
    df["output"] = (df["output"]  - df["output"].min()) /( df["output"].abs().max() - df["output"].min())

        #print(df)

        #print(len(df["output"]))
        #print(len(df["x"]))
        #print(len(df["y"]))




        #fig=plt.figure()
        #ax=fig.add_subplot(111,projection='3d')

        #ax.scatter(df["x"],df["y"],df["output"],color="red")
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        #ax.set_zlabel("dist from rt")

        #plt.show()
    print(metric_name)
    #print(df[["x", "y"]])
    print(regressionGrad(df[["x","y"]],df["output"] ))


    
    '''
    #getCorrelations(metric_list, p)         
        #fig = generate_3DBarChart(arr.astype(float), np.array(axis1).astype(float), np.array(axis2).astype(float))
        
    #generate_MultiVar_heatmap(arr.astype(float), np.array(axis1).astype(float), np.array(axis2).astype(float))

    


