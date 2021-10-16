#!/usr/bin/env python3
"""
Copyright 2019 Marco Lattuada

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
import datetime
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from xml.dom.minidom import parseString

import dicttoxml
import numpy
import xmltodict

import app

def main():
    """
    The wrapper script for rhd case study

    The parameters are:
        -d, --debug: enables the printing of the debug messages
        -p, --parameters: a comma-separated list of parameters to be passed to the wrapped application
    """
    #The absolute path of the current script
    abs_script = os.path.abspath(sys.argv[0])

    #The root directory of the script
    abs_root = os.path.dirname(abs_script)

    sys.path.append(os.path.join(abs_root, ".."))
    utility = __import__("utility")

    #Parsing input arguments
    parser = argparse.ArgumentParser(description="Train the network for rhd")
    parser.add_argument('-p', "--parameters", help="Parameters to be overwritten", required=True)
    parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    args = parser.parse_args()

    #Initializing logger
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #Root tree node
    root = compute_parameters(args.parameters)

    #Get input_data_path
    input_data = root.get("input_data")
    if not input_data:
        logging.error("input_data not set in configuration file")
        sys.exit(1)

    #Get number of epochs
    num_epochs = root.get("num_epochs")
    if not num_epochs:
        logging.error("num_epochs not set in configuration file")
        sys.exit(1)

    #Get batch size
    batch_size = root.get("batch_size")
    if not batch_size:
        logging.error("batch_size not set in configuration file")
        sys.exit(1)

    #Get number of images
    num_images = root.get("num_images")
    if not num_images:
        logging.error("num_images not set in configuration file")
        sys.exit(1)

    #Get GPUs number
    gpus_number = root.get("gpus_number")
    if not gpus_number:
        logging.error("gpus_number not set in configuration file")
        sys.exit(1)

    full_labels_file_name = os.path.join(input_data, "rhd_preprocessed", "instances_info.csv")
    full_labels_file = open(full_labels_file_name, "r")
    labels_file_name = os.path.join(tempfile.gettempdir(), "instances_info_" + num_images + ".csv")
    labels_file = open(labels_file_name, 'w')
    for line in itertools.islice(full_labels_file, int(num_images)):
        labels_file.write(line)

    full_labels_file.close()
    labels_file.close()

    logging.info("Created %s", labels_file_name)

    ###Adding rhd version (hash of github commit)
    root["rhd_version"] = "a93e1ed"

    #Adding system UUID
    if not os.path.exists("/etc/machine-id"):
        logging.warning("/etc/machine-id does not exists")
    else:
        uuid_line = open("/etc/machine-id", "r").readline()
        uuid = uuid_line

        root["system_UUID"] = uuid

    #Dump configuration in xml
    dicttoxml.set_debug(False)
    generated_xml = parseString(dicttoxml.dicttoxml(root, custom_root="rhd_configuration", attr_type=False)).toprettyxml(indent="   ")
    generated_xml_file = open("configuration.xml", "w")
    generated_xml_file.write(generated_xml)
    generated_xml_file.close()

    project_root = utility.get_project_root()

    rhd_command = "python3 " + os.path.join(project_root, "apps", "rhd", "rhd.py") + " --data-path '" + os.path.join(input_data, "rhd_preprocessed", "final_dataset") + "' --models-path '" + os.path.join(input_data, "rhd-classification", "models") + "' --results-path '" + os.path.join(input_data, "rhd-classification", "results") + "' --labels-file-path '" + labels_file_name + "' --doppler-filtering 'none' --undersampling-filtering 'none' --learning-rate 1e-4 --num-epochs " + num_epochs + " --batch-size " + batch_size + " -tl --multi-gpu " + gpus_number
    logging.info("rhd command is %s", rhd_command)
    starting_time = time.time()
    return_value = subprocess.call(rhd_command, shell=True, executable="/bin/bash")
    ending_time = time.time()
    if return_value:
        logging.error("Error in execution of %s", rhd_command)
        sys.exit(1)
    stdout_file_name = os.path.join(os.getcwd(), "execution_stdout")
    stdout_file = open(stdout_file_name, "r")
    for line in stdout_file:
        if "Time logs per batch per epoch will be saved to" in line:
            path = line.replace("\n", "").split()[-1]
            for epoch in range(0, int(num_epochs)):
                iteration_file_name = os.path.join(path, str(epoch) + ".npy")
                shutil.copy(iteration_file_name, os.getcwd())
            break
    #Dump information about experiment
    print("rhd training")
    print("Number of images is " + str(num_images))
    print("Overall training time is " + str(ending_time - starting_time) + " seconds (" + time.ctime(starting_time) + " ==> " + time.ctime(ending_time) + ")")
    print("Configuration file is:")
    print(generated_xml)


def compute_parameters(cl_parameters):
    """
    Compute the configuration name on the basis of the values of the experiment parameters

    Paramters
    ---------
    cl_parameters: str
        A comma separated list of parameter=value

    Return
    ------
    str
        The configuration name
    """
    parameters = app.load_xml_configuration(cl_parameters, "rhd", "rhd_configuration")
    #Overwrite parameters
    for cl_parameter in cl_parameters.split(","):
        tokens = cl_parameter.split("=")
        if len(tokens) != 2:
            logging.error("parameters must be a , seperated list of <parameter>=<value>")
            sys.exit(1)
        if not tokens[0] in parameters and tokens[0] not in {"configuration", "gpus_number", "gpus_instances"}:
            logging.error("parameter %s is not present in the source configuration", tokens[0])
            sys.exit(1)
        parameters[tokens[0]] = tokens[1]
    return parameters

def collect_data(repetition_path, gpu_type, gpus_number, debug):
    """
    Add to csv (and creates it if it does not exist) data about the experiment whose output was saved in repetition_path

    Parameters
    ----------
    repetition_path: str
        The path containing the output of the currently analyzed experiment

    gpu_type: str
        The type of the GPU

    gpu_number: str
        The number of the GPUs of the VM

    debug: boolean
        True if debug messages have to be printed
    """
    csv_file_name = "rhd.csv"
    if os.path.exists(csv_file_name):
        csv_file = open(csv_file_name, "a")
    else:
        csv_file = open(csv_file_name, "w")
        column_labels = [
            "starting timestamp",
            "starting time",
            "rhd version",
            "system_UUID",
            "mac_address",
            "vm_instance",
            "GPU type",
            "GFlops",
            "disk speed",
            "GPU number",
            "Images Number",
            "batch size",
            "Real Iterations Number",
            "epochs",
            "training time",
            "overall execution time",
            "missing time",
            "repetition number",
            "path"
            ]
        index = 0
        for column_label in column_labels:
            if index != 0:
                csv_file.write(",")
            csv_file.write(column_label + "(" +str(index) + ")")
            index = index + 1
        csv_file.write("\n")

    #Getting information from execution_stdout
    execution_stdout_filename = os.path.join(repetition_path, "execution_stdout")
    execution_stdout = open(execution_stdout_filename)
    starting_timestamp = ""
    for line in execution_stdout:
        if line.startswith("Overall training time is"):
            overall_execution_time = line.split()[4]
            times_vector = re.split(r'\(|=|>|\)', line)
            starting_timestamp = int((datetime.datetime.strptime(times_vector[1].strip(), "%a %b %d %H:%M:%S %Y") - datetime.datetime(1970, 1, 1)).total_seconds())
            starting_time = (datetime.datetime.strptime(times_vector[1].strip(), "%a %b %d %H:%M:%S %Y")).strftime("%Y-%m-%d %H:%M:%S")
    execution_stdout.close()

    if starting_timestamp == "":
        csv_file.close()
        return

    xml_configuration_file_name = os.path.join(repetition_path, "configuration.xml")
    if not os.path.exists(xml_configuration_file_name):
        csv_file.close()
        return

    xml_configuration = xmltodict.parse(open(xml_configuration_file_name).read(), force_list={'input_class'})["rhd_configuration"]

    utility = __import__("utility")

    #Getting information from hw configuration
    mac_address = ""
    machine_name = ""
    for line in open(os.path.join(repetition_path, "hw_configuration")):
        if line.find("serial: ") != -1:
            mac_address = line.split()[1]
        if line.startswith("Linux "):
            machine_name = line.split()[1]


    #Retrieving machine information
    #Add host_scripts to the directories for python modules search
    host_scripts_path = os.path.join(utility.get_project_root(), "host_scripts")
    sys.path.append(host_scripts_path)
    collect_data_module = __import__("collect_data")


    rhd_version = xml_configuration["rhd_version"]
    system_uuid = xml_configuration["system_UUID"]
    machine_information = collect_data_module.get_machine_information(mac_address, machine_name, system_uuid)
    mac_address = machine_information["mac_address"]
    system_uuid = machine_information["system_uuid"]
    machine_name = machine_information["machine_name"]
    gflops = machine_information["gflops"]
    disk_speed = machine_information["disk_speed"]
    repetition_number = os.path.basename(repetition_path)

    if "gpus_number" in xml_configuration:
        gpus_number = xml_configuration["gpus_number"]

    num_images = xml_configuration["num_images"]
    batch_size = xml_configuration["batch_size"]
    num_epochs = xml_configuration["num_epochs"]

    #Getting information from *.npy
    training_time = 0
    real_iterations_number = 0
    for epoch in range(0, int(num_epochs)):
        iteration_file_name = os.path.join(repetition_path, str(epoch) + ".npy")
        iteration_data = numpy.load(iteration_file_name)
        real_iterations_number = real_iterations_number + iteration_data.size
        for iteration_time in iteration_data:
            training_time = training_time + iteration_time

    missing_time = float(overall_execution_time) - training_time

    line = [
        starting_timestamp,
        starting_time,
        rhd_version,
        system_uuid,
        mac_address,
        machine_name,
        gpu_type,
        gflops,
        disk_speed,
        gpus_number,
        num_images,
        batch_size,
        real_iterations_number,
        num_epochs,
        training_time,
        overall_execution_time,
        missing_time,
        repetition_number,
        repetition_path
    ]
    csv_file.write(",".join([str(token) for token in line]) + "\n")
    csv_file.close()

def compute_configuration_name(cl_parameters):
    """
    Compute the configuration name on the basis of the values of the experiment parameters

    Paramters
    ---------
    cl_parameters: str
        A comma separated list of parameter=value

    Return
    ------
    str
        The configuration name
    """
    parameters = compute_parameters(cl_parameters)
    configuration_name = "ni_" + parameters["num_images"] + "_ep_" + parameters["num_epochs"] + "_bs_" + parameters["batch_size"]
    if "gpus_number" in parameters:
        gpus_number = "_gpus_number_" + parameters["gpus_number"]
    else:
        gpus_number = ""
    configuration_name = configuration_name + gpus_number

    return configuration_name

if __name__ == "__main__":
    main()
