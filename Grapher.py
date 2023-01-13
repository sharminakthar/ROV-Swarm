import os
import shutil
from pathlib import Path
from textwrap import wrap
import itertools
import pandas as pd
import numpy as np
import seaborn as sb
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

from MetricsList import units_list, metric_list
from metrics.BaseMetric import BaseMetric
from metrics.Helper_Functions import moving_average
from DoubleAxisGrapher import DAG
from metrics.Heatmaps import GetHeatmap

class Grapher():
    
    #runs selected graphing functions based on user input
    def __init__(self, parameters: dict, outputFunc):
        if outputFunc == double_var_charts:
            double_var_charts(self, parameters["path"], parameters["metrics"])
        elif outputFunc == doubleAxisGraph:
            doubleAxisGraph(self, parameters["path"], parameters["mission"], parameters["single_var"], parameters["metric1"], parameters["metric2"], parameters["run"], parameters["errors"] )
        elif outputFunc == heatmaps:
            heatmaps(self, parameters["path"], parameters["mission"], parameters["single_var"], parameters["run"], parameters["errors"])
        elif outputFunc == correlationCoefficients:
            correlationCoefficients(self, parameters["path"], parameters["metrics"], parameters["vars"])
        elif outputFunc == double_var_regress:
            double_var_regress(self, parameters["path"], parameters["metrics"], parameters["regr_x1"], parameters["regr_x2"], parameters["regr_metric"])
        elif outputFunc == single_var_charts:
            
            single_var_charts(self, parameters["path"], parameters["metrics"], parameters["vars"])


    #retrieves data of single value from single variable
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
        for i, run in enumerate(dirs):
            run = run / "raw_data_log.csv"
            data = pd.read_csv(run)
            result = metric.calculate(data)
            if i == 0:
                results["Timestep"] = result.iloc[:, 0]
            results[str(i)] = result.iloc[:, 1]
    
        return pd.DataFrame(results)

    #retrieves data of all values of a variables
    def get_single_var_data(self, metric: BaseMetric, directory: Path, reduction="mean") -> dict[str, pd.DataFrame]:
        """
        Takes in a metric and a directory of a single variable and produces the metric data 
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



    #retrieves data of all values of all variables
    def generate_data(self, metric_list: dict, directory: Path, reduction: str="mean", vars = None):
        data_folder = directory / "Metric_Data"

        # Find all folders two up from the raw data - i.e. the list of 1d variables to run
        final_var_dirs = []
        for dirpath, dirnames, filenames in os.walk(directory):
            if "Metric_Data" in dirpath or "Graphs" in dirpath or "raw_data_log.csv" in filenames or dirpath or any(var in dirpath for var in vars) == False:
                continue
            if "metadata.csv" in filenames:
                parent = os.path.dirname(dirpath)
                if parent not in final_var_dirs: 
                    final_var_dirs.append(parent)
        
        for d in final_var_dirs:
            p = Path(d)
            for metric_name, metric_info in metric_list.items():
                name = "{} - {}".format(p.name, metric_name)
                print(name)
                if len(os.listdir(p)) > len(os.listdir(data_folder / p.name / metric_name)):
                    #ind_var_output = self.get_single_var_data(metric, var, reduction=reduction)
                    folder = data_folder / p.name / metric_name    
                    metric = metric_info["instance"]
                    #folder = data_folder / d[len(str(directory))+1:] / metric_name
                    if metric_name == "col_num":
                        reduction = "none"
                    ind_var_output = self.get_single_var_data(metric, p, reduction=reduction)
                    
                # k is the value of the independent variable
                # v is the metric data at that variable
                    for k, v in ind_var_output.items():
                        f = folder / k 
                        f.mkdir(parents=True, exist_ok=True)
                        v.to_csv(path_or_buf=(f / "metric_data.csv"), index=False)

    
    #plots selected graph on each independent variable for each metric
    def graph_single_var(self, data_path: Path, metric_list: dict, graph_func, save_folder: str = None, vars = None, **kwargs):
        """
        Will graph all metrics on a single variable
        """

        for var in data_path.iterdir():
            if (vars != None and (var.name in vars )== False or (vars == None)):
                for metric_name, metric_info in metric_list.items():

                    print(var.name + ": " + metric_name)
                    ind_var_output = {}
                    folder = var / metric_name 
                    if not folder.exists():
                        continue
                    for k in folder.iterdir():
                        df = pd.read_csv(k / "metric_data.csv")
                        ind_var_output[k.name] = df
                
                    fig = graph_func(self, ind_var_output, metric_info, var.name, **kwargs)
                    if save_folder !=  None:
                        folder = data_path.parent / save_folder / var.name / metric_name
                    else:
                        folder = data_path.parent / "Graphs" / var.name / metric_name

                    folder.mkdir(parents=True, exist_ok=True)
                    fig.savefig(folder / (metric_name + ".png"), bbox_inches="tight")
                    plt.close(fig)
                    


    #plots selected graph on two co-varting independent variables for each metric
    def graph_double_var(self, data_path: Path, metric_list: dict, graph_func, save_folder: str, **kwargs):
        """
        Will graph all metrics on two variables that are co-varying
        """
        axis1 = [n.name for n in data_path.iterdir()]
        axis2 = [n.name for n in next(data_path.iterdir()).iterdir()]

        for i,var1 in enumerate(data_path.iterdir()):
            for j,var2 in enumerate(var1.iterdir()):
                for metric_name, metric_info in metric_list.items():
                    ind_var_output = {}
                    folder = var1 / metric_name 
                    if not folder.exists():
                        continue
                    for k in folder.iterdir():
                        df = pd.read_csv(k / "metric_data.csv")
                        ind_var_output[k.name] = df
                    
                    fig = graph_func(ind_var_output, metric_info, var.name, **kwargs)
                    folder = data_path.parent / save_folder / var.name / metric_name
                    folder.mkdir(parents=True, exist_ok=True)
                    fig.savefig(folder / (metric_name + ".png"), bbox_inches="tight")
                    plt.close(fig)

    
    #generates moving line chart - each value is average of adjacent points next to it for single independent variable
    def generate_smooth_line_chart(self, data, metric_info, var, adj_points=3):

        var_name = " ".join([w.capitalize() for w in var.split("_")])
        fig = plt.figure()
        ax = fig.add_subplot()
        labels = sorted(list(data.keys()), key=lambda x: float(x))
        for k in labels:

            #second parameter is the number adjacent values each data point is averaged with
            k2 = moving_average(data[k], adj_points)

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


    #generates non-smoothed line chart for single independent variable
    def generate_line_chart(self, data, metric_info, var):
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


    #generates bar chart for single independent variable
    def generate_bar_chart(self, data, metric_info, var, bar_reduction="mean"):
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
        axis1_index = {n:i for i,n in enumerate(axis1)}
        axis2 = [n.name for n in next(next(data_path.iterdir()).iterdir()).iterdir()]
        axis2_index = {n:i for i,n in enumerate(axis2)}
        
        data_arr = np.zeros((len(axis1), len(axis2)))
        for packet_loss in data_path.iterdir():
            for metric in packet_loss.iterdir():
                if metric.name == metric_name:
                    for bandwidth in metric.iterdir():
                        data = pd.read_csv(bandwidth / "metric_data.csv")
                        data_arr[axis1_index[packet_loss.name], axis2_index[bandwidth.name]] = reduction(data.iloc[:,1].to_numpy())
        
        return axis1, axis2, data_arr 


    
def multivar_grapher( metric_list: dict, directory: Path, reduction: str="mean", graph_func1=None, graph_func2=None, graph_func3=None):
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


#bar reduction converts data of single value set for variable over the simulation into one data point. 
#input string determines how it is converted
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

#generates heatmap of two co-varying independent variables
def generate_multivar_heatmap( dataArray, xlabels, ylabels):
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

#generates 3D bar chart of co-varying independent variables
def generate_3Dbar( dataArray, ylabels, xlabels):
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

#generates surface plot of two co-varying independent variables
def generate_surface_plot(dataArray, ylabels, xlabels):
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


# retrieves correlation coefficients of selected variables on selected metrics
def getCorrelations( metric_list: dict, directory: Path, varList = None):


    data_path = directory/"Metric_Data"
    numOfMetrics = len(metric_list)
    allCoeffs = np.empty([0,numOfMetrics])
    vars = []

    for var in data_path.iterdir():
        if varList != None and (var.name in varList) == False:
            continue
        vars.append(var.name)
        coeffs = []
        metrics = []

        for metric in var.iterdir():    
            if (metric.name in metric_list) == False:
                continue
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
                
     
            coeffs.append(determineCoefficient(labels, vals))

        if len(coeffs) != len(metric_list):
            print("MISSING METRIC")
            print(coeffs)
        else:
            allCoeffs = np.vstack([allCoeffs, coeffs])

    allCoeffs = zip(*allCoeffs)

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

        print("")

#used in getCorrelations to sort one list w.r.t another
def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs)]
 
    return z 


#performs spearman rank correlation
def determineCoefficient(d1, d2):
    corr, _ = spearmanr(d1,d2)
    return corr

#performs linear regression on two co-varying variables, to predict output metric value given a pair of these variable values
def regressionGrad(self, X, y, reg_x1,reg_x2):
    #x - xlabels and y labels
    #y = metric
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    arr = []
    for i in range(len(reg_x1)):
    
        predicted = regr.predict([[reg_x1[i], reg_x2[i]]])
        arr.append(predicted)
    return arr

#calls and generates all graphs for co-varying independent variables
def double_var_charts(p: Path, metric_list):  
    Grapher.generate_data(metric_list, p)
    Grapher.graph_double_var(p / "Metric_Data", metric_list, Grapher.generate_multivar_heatmap)
    Grapher.graph_double_var(p / "Metric_Data", metric_list, Grapher.generate_3D_bar)
    Grapher.graph_double_var(p / "Metric_Data", metric_list, Grapher.generate_surface_plot)
    double_var_regress(p / "Metric_Data", metric_list)

#runs double-axis grapehr
def doubleAxisGraph(self, p, mission, var, metric1, metric2, run, errors ):
    Grapher.generate_data(self, metric_list, p, vars)

    b = DAG(p, mission, var, metric1, metric2, run, errors)

#runs heatmap grapher
def heatmaps(self, p, mission, var, run, errors):

    GetHeatmap(p,mission,var,run, errors)


#retrieves correlation coefficients and prints them in terminal
def correlationCoefficients(self, p: Path, metric_list, vars = None):
    Grapher.generate_data(self, metric_list, p, vars)
    getCorrelations(metric_list, p, vars)


#runs regression on two co-varying independent variables - outputs in terminal
def double_var_regress(self, p: Path, metric_list, reg_x1, reg_x2, metric):
    metric_name = metric
    axis1, axis2, arr = read_multi_data(p, metric_name)


    print(arr)
    c = list(itertools.product(axis1, axis2))
    output = arr.flatten()
    c1 = list(zip(*c))[0]
    c2 = list(zip(*c))[1]

    df = pd.DataFrame({'x':c1, 'y':c2})

    df["output"] = output
    #normalize if putting values between 0 and 1
    #df["output"] = (df["output"]  - df["output"].min()) /( df["output"].abs().max() - df["output"].min())
    

    print("x1 values: " , reg_x1)
    print("x2 values: ", reg_x2)
    print("Predicted " + metric + " values:")
    print(regressionGrad(self, df[["x","y"]],df["output"], reg_x1, reg_x2 ))

#runs line and bar charts for selected independent variables on selected metrics
def single_var_charts(self, p: Path, metric_list, vars):
    Grapher.generate_data(self, metric_list, p, vars = vars)
    Grapher.graph_single_var(self, p / "Metric_Data",metric_list, Grapher.generate_line_chart, vars = vars)
    Grapher.graph_single_var(self, p / "Metric_Data", metric_list, Grapher.generate_bar_chart, vars = vars)        


if __name__ == "__main__":

    #path of data folder
    p = Path("out/RACETRACK_MULTI_COMM_FLOCK_SIZE\FLOCK_SIZE-PACKET_LOSS")


    #selected output function
    outputs = {
        #co-varying independent variable grapher
        '1' : double_var_charts,

        #generate line graphs for variable with two different metrics
        '2' : doubleAxisGraph,

        #generate heatmaps of swarm paths
        '3' : heatmaps,

        #generate correlation coefficients of each variable to each metric
        '4' : correlationCoefficients,

        #generate projected value of two associated independent variables for a given metric
        '5' : double_var_regress,

        #bar and line charts for single independent variable data
        '6' : single_var_charts
    }

    #Vars to analyse if not all in directory, otherwise set as None in parameters
    vars = ["FLOCK_SIZE", "BANDWIDTH", "PACKET_LOSS", "SPEED_ERROR", "HEADING_ERROR", 
                 "RANGE_ERROR", "BEARING_ERROR", "ACCELERATION_ERROR", "SPEED_CALIBRATION_ERROR",
                 "HEADING_CALIBRATION_ERROR", "RANGE_CALIBRATION_ERROR", "BEARING_CALIBRATION_ERROR",
                 "ACCELERATION_CALIBRATION_ERROR"]
    #vars

    parameters = {

        #directory of data
        'path' : p,
        
        
        #metrics to be analyse, see MetricList.py
        'metrics' : metric_list,
        
        
        #vars to analyse, leave as None for all vars in directory
        #'vars' : None,
        'vars' : ["HEADING_CALIBRATION_ERROR", "BEARING_CALIBRATION_ERROR"],
        
        #mission used for data collection - used only for heatmaps grapher
        'mission' : "RACETRACK",
        #'mission' : "FOLLOW_CIRCLE",
        #'mission' : "FIXED_HEADING",
        
        #single var to analyse - used only in Double axis or Heatmap grapher
        'single_var' : "HEADING_CALIBRATION_ERROR",

        #run instance if using double axis or heatmap function
        'run' : 0,

        #metrics to compare using double axis graph
        'metric1' : "distfromRT",
        'metric2' : "sep_mean",

        #error values to plot using double axis or heatmap grapher. Leave as None if selecting all values of variable in directory
        "errors" : ["0", "30", "10", "40"],
        #"errors" : None 

        #pairs of values from two independent variables to predict output metric value of
        'regr_x1' : [0,1,2,3],
        'regr_x2' : [10,12,14,16],

        #metric to calculate regression
        'regr_metric' : "distfromRT"
    }


    grapher = Grapher(parameters, outputs["5"])

    








    #grapher.generate_data(metric_list, p)

    quit()
    grapher.graph_single_var(Path("out/FIXED_HEADING_ULTRA_EXTENDED/Metric_Data"), metric_list, grapher.generate_line_chart,
                             "Graphs")



 















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

    # grapher.get_all_var_data(metric_list, p, save_folder="line", load_data=False, graph_func=Grapher.generate_line_chart)
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
   


