#!/usr/bin/env python3
"""
Copyright 2018 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import logging
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Collect experiment results")
parser.add_argument("root_directory", help="The root directory containing the results to be processed")
parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
parser.add_argument('-i', "--interval", help="The interval to be considered (i.e., experiment run outside interval are excluded from generated csv")
parser.add_argument('-b', "--add-blacklisted", help="Add also the blacklisted experiments to the generated csvs", default=False, action="store_true")

args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#The absolute path of the current script
abs_script = os.path.abspath(sys.argv[0])

#The root directory of the script
abs_root = os.path.dirname(abs_script)

#The script for sorting csv
sort_script = os.path.join(abs_root, "sort_csv.py")

#Add apps to the directory for python packages search
apps_path = os.path.join(abs_root, "..", "apps")
logging.debug("Adding %s to sys paths", apps_path)
sys.path.append(apps_path)

#Absoute path of directory containing the data
abs_root_data = os.path.abspath(args.root_directory)

#If there is any csv in the current directory aborts; user must delete them
for local_file in os.listdir(os.getcwd()):
    extension = os.path.splitext(local_file)[1]
    if extension == ".csv":
        logging.error("Found csv file in current directory: %s", local_file)
        sys.exit(1)

if args.interval != None:
    if args.interval.find(":") != -1:
        split = args.interval.split(":")
        interval_begin = split[0]
        interval_end = split[1]
    else:
        interval_begin = args.interval
        interval_end = args.end
    interval_begin = interval_begin.rstrip()
    interval_end = interval_end.rstrip()
    #For the moment interval boundaries can only be date
    if interval_begin.find(" ") != -1:
        logging.error("whitespace in interval begin")
        sys.exit(-1)
    if interval_end.find(" ") != -1:
        logging.error("whitespace in interval end")
        sys.exit(-1)
    #Adding time to the interval boundaries"
    interval_begin = interval_begin + "_00-00-00"
    interval_end = interval_end + "_23_59_59"

for vm in os.listdir(abs_root_data):
    vm_path = os.path.join(abs_root_data, vm)
    if os.path.isdir(vm_path):
        gpu_number = 0
        gpu_type = ""
        if vm == "iruel":
            gpu_number = 0
            gpu_type = "-"
        elif vm == "StandardB1ms":
            gpu_number = 0
            gpu_type = "-"
        elif vm == "StandardNC6":
            gpu_number = 1
            gpu_type = "K80"
        elif vm == "StandardNC12":
            gpu_number = 2
            gpu_type = "K80"
        elif vm == "StandardNC24":
            gpu_number = 4
            gpu_type = "K80"
        elif vm == "StandardNV6":
            gpu_number = 1
            gpu_type = "M60"
        elif vm == "StandardNV12":
            gpu_number = 2
            gpu_type = "M60"
        elif vm == "StandardNV24":
            gpu_number = 4
            gpu_type = "M60"
        elif vm == "matemagician":
            gpu_number = 2
            gpu_type = "Quadro P600"
        elif vm == "ubuntu-xenial":
            gpu_number = 0
            gpu_type = "-"
        elif vm == "Mine":
            continue
        else:
            logging.error("%s is not a known name of VM", vm)
            sys.exit(-1)
        for app in os.listdir(vm_path):
            app_path = os.path.join(vm_path, app)
            if os.path.isdir(app_path) and os.path.exists(os.path.join(apps_path, app + ".py")):
                app_package = __import__(app)
                for experiment_configuration in os.listdir(app_path):
                    experiment_configuration_path = os.path.join(app_path, experiment_configuration)
                    if os.path.isdir(experiment_configuration_path):
                        for experiment in os.listdir(experiment_configuration_path):
                            #Check if experiment is in the interval
                            if args.interval != None:
                                #Because of the format of the timestamp, they are actually sortable strings
                                if experiment < interval_begin or experiment > interval_end:
                                    continue
                            experiment_path = os.path.join(experiment_configuration_path, experiment)
                            if os.path.isdir(experiment_path):
                                for repetition in os.listdir(experiment_path):
                                    repetition_path = os.path.join(experiment_path, repetition)
                                    if os.path.isdir(repetition_path):
                                        if os.path.exists(os.path.join(repetition_path, "skip")):
                                            continue
                                        logging.debug("Processing directory %s", repetition_path)
                                        app_package.collect_data(repetition_path, gpu_type, gpu_number, args.debug)
                sort_command = sort_script + " -i" + app + ".csv -o" + app + ".csv -c0"
                subprocess.call(sort_command, shell=True, executable="/bin/bash")
