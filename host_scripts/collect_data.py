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
import csv
import logging
import os
import subprocess
import sys

machine_data = {}

def get_machine_information(mac_address, machine_name="", system_uuid=""):
    """
    Return information about a (virtual) machine

    Parameters
    ----------
    mac_address: str
        The mac address of the machine

    machine_name: str
        The hostname of the machine

    system_uuid: str
        The UUID of the machine

    Return
    ------
    dict of str: object
        A dictionary containing the information of the identified machine"
    """
    global machine_data

    #Load machine data only once
    if not machine_data:

        #The absolute path of the current script
        abs_script = os.path.realpath(__file__)

        #The root directory of the script
        abs_root = os.path.dirname(abs_script)

        #Provider directory
        providers_dir = os.path.join(abs_root, "..", "providers")
        for machine_information in os.listdir(providers_dir):
            if machine_information.endswith("machine_information.csv"):
                data = csv.reader(open(os.path.join(providers_dir, machine_information)))
                for line in data:
                    temp_mac_address = line[0]
                    machine_data[temp_mac_address] = {}
                    machine_data[temp_mac_address]["mac_address"] = line[0]
                    machine_data[temp_mac_address]["system_uuid"] = line[1]
                    machine_data[temp_mac_address]["machine_name"] = line[2]
                    machine_data[temp_mac_address]["gflops"] = line[3]
                    machine_data[temp_mac_address]["disk_speed"] = line[4]

    if mac_address != "":
        if mac_address in machine_data:
            return machine_data[mac_address]
        logging.error("Information about machine with mac %s not available", mac_address)
        sys.exit(1)
    if machine_name != "":
        for search_mac_address in machine_data:
            if machine_data[search_mac_address]["machine_name"] == machine_name:
                return machine_data[search_mac_address]
        logging.error("machine named %s not found", machine_name)
        sys.exit(1)
    logging.error("mac address or machine name must be provided")
    sys.exit(1)

def main():
    """
    Script for collecting profiling data.

    This script creates a CSV file for each application. Existing scripts cannot be overwritten, so the script fails if it finds in the current directory a CSV file.

    Parameters of the scripts are:
    root_directory: the root directory which is analyzed to collect profiling information. Data must be organized in a directories hierarchy with structure <hostname>/<app>/<configuration_name>/<timestamp>/<repetition_number>
    -d, --debug: enables the debug printing
    -i, --interval: the timestamp interval to be considered in profiling data collection. Experiments outside this interval are ignored
    -b, --add-blacklisted: experiments which are blacklisted (i.e., the corresponding directory contains a file named skip) are included in the generated CSV file
    -a, --app: generates CSV file only for a single application
    """
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument("root_directory", help="The root directory containing the results to be processed")
    parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    parser.add_argument('-i', "--interval", help="The interval to be considered (i.e., experiment run outside interval are excluded from generated CSV")
    parser.add_argument('-b', "--add-blacklisted", help="Add also the blacklisted experiments to the generated CSVs", default=False, action="store_true")
    parser.add_argument('-a', "--app", help="The app whose data have to be collected (default: all")

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

    #Add apps to the directory for python modules search
    apps_path = os.path.join(abs_root, "..", "apps")
    logging.debug("Adding %s to sys paths", apps_path)
    sys.path.append(apps_path)

    sys.path.append(os.path.join(abs_root, ".."))

    #Absolute path of directory containing the data
    abs_root_data = os.path.abspath(args.root_directory)

    #If there is any csv in the current directory aborts; user must delete them
    if args.app:
        if os.path.exists(args.app + ".csv"):
            logging.error("Found %s in current directory", args.app + ".csv")
            sys.exit(1)
    else:
        for local_file in os.listdir(os.getcwd()):
            extension = os.path.splitext(local_file)[1]
            if extension == ".csv":
                logging.error("Found csv file in current directory: %s", local_file)
                sys.exit(1)

    if args.interval:
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

    for machine_name in os.listdir(abs_root_data):
        vm_path = os.path.join(abs_root_data, machine_name)
        machine_name = machine_name.split('-')[0]
        if os.path.isdir(vm_path):
            gpu_number = 0
            gpu_type = ""
            if machine_name == "bardiel":
                gpu_number = 1
                gpu_type = "GeForce GTX 1050"
            elif machine_name == "iruel":
                gpu_number = 0
                gpu_type = "-"
            elif machine_name in {"StandardB1ms", "standardb1S"}:
                gpu_number = 0
                gpu_type = "-"
            elif machine_name in {"StandardNC6", "standardnc6"}:
                gpu_number = 1
                gpu_type = "K80"
            elif machine_name == "StandardNC12":
                gpu_number = 2
                gpu_type = "K80"
            elif machine_name == "StandardNC24":
                gpu_number = 4
                gpu_type = "K80"
            elif machine_name == "StandardNV6":
                gpu_number = 1
                gpu_type = "M60"
            elif machine_name == "StandardNV12":
                gpu_number = 2
                gpu_type = "M60"
            elif machine_name == "StandardNV24":
                gpu_number = 4
                gpu_type = "M60"
            elif machine_name == "matemagician":
                gpu_number = 2
                gpu_type = "Quadro P600"
            elif machine_name == "ubuntu-xenial":
                gpu_number = 0
                gpu_type = "-"
            elif machine_name in {"d2beb36896eb2021", "bb657a076dc92021"}:
                gpu_number = 1
                gpu_type = "GTX 1080Ti"
            elif machine_name == "asus-PC":
                gpu_number = 1
                gpu_type = "GeForce GT 750M"
            elif machine_name == "polimi-gpu-trial":
                gpu_number = 4
                gpu_type = "Tesla P100"
            elif machine_name in {"dgxstation-ita", "ada9167d4635", "e03c695a45a1"}:
                gpu_number = 4
                gpu_type = "Tesla V100"
            elif machine_name in {"upvdocker1GPU"}:
                gpu_number = 1
                gpu_type = "Tesla V100"
            elif machine_name == 'gio-XPS-15-9560':
                gpu_number = 1
                gpu_type = "GeForce GTX 1050"    
            else:
                logging.warning("%s is not a known name", machine_name)
                continue
            for app in os.listdir(vm_path):
                app_path = os.path.join(vm_path, app)
                if os.path.isdir(app_path) and os.path.exists(os.path.join(apps_path, app + ".py")) and (not args.app or args.app == app):
                    app_module = __import__(app)
                    for experiment_configuration in os.listdir(app_path):
                        experiment_configuration_path = os.path.join(app_path, experiment_configuration)
                        if os.path.isdir(experiment_configuration_path):
                            for experiment in os.listdir(experiment_configuration_path):
                                #Check if experiment is in the interval
                                if args.interval:
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
                                            app_module.collect_data(repetition_path, gpu_type, gpu_number, args.debug)
                    if not os.path.exists(app + ".csv"):
                        continue
                    if app in {"tf", "pytorch"}:
                        sort_command = sort_script + " -i" + app + ".csv -o" + app + ".csv -c0,14"
                    else:
                        sort_command = sort_script + " -i" + app + ".csv -o" + app + ".csv -c0"
                    subprocess.call(sort_command, shell=True, executable="/bin/bash")

if __name__ == "__main__":
    main()
