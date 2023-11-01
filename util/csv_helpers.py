from datetime import datetime
import os.path
import csv
from flock_settings import Setting


def get_new_output_directory():
    current_directory = os.getcwd()    
    output_dir = os.path.join(current_directory, "out")

    new_dir = datetime.now().strftime("%Y-%m-%dT%H_%M")
    dir = os.path.join(output_dir, new_dir)

    if not os.path.exists(dir):
        os.makedirs(dir)
        return dir

    index = 2
    dir_enumerated = dir + "_2"

    while os.path.exists(dir_enumerated):
        index += 1
        dir_enumerated = dir + "(" + str(index) + ")"

    os.makedirs(dir_enumerated)
    return dir

def output_dataframe_csv(dataframe, directory, name):
    path = os.path.join(directory, name)
    dataframe.to_csv(path_or_buf=path, index=False)

def output_metrics_log(simulator, directory):
    output_dataframe_csv(simulator.get_metrics_log(), directory, "metrics_log.csv")

def output_raw_data_log(simulator, directory):
    output_dataframe_csv(simulator.get_raw_data_log(), directory, "raw_data_log.csv")

def output_metadata(settings, directory):
    path = os.path.join(directory, "metadata.csv")

    file = open(path, 'w', newline='', encoding='utf-8')

    writer = csv.writer(file)

    writer.writerow(["Setting", "Value"])

    for (name, setting) in Setting.__members__.items():
        value = settings.get(setting)

        writer.writerow([name, value])

    file.close()