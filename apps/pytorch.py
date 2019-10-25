#!/usr/bin/python3
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
import datetime
import linecache
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from xml.dom.minidom import parseString
import xmltodict

import dicttoxml

def compute_parameters(cl_parameters):
    configuration_base = "default"
    #First look for configuration
    for cl_parameter in cl_parameters.split(","):
        if len(cl_parameter.split("=")) != 2:
            logging.error("parameters must be a , seperated list of <parameter>=<value>: %s", cl_parameter)
            sys.exit(1)
        if cl_parameter.split("=")[0] == "configuration":
            configuration_base = cl_parameter.split("=")[1]
            break

    #Load configuration
    parameters = load_xml_configuration(configuration_base + ".xml")["pytorch_configuration"]

    #Overwrite parameters
    for cl_parameter in cl_parameters.split(","):
        tokens = cl_parameter.split("=")
        if len(tokens) != 2:
            logging.error("parameters must be a , seperated list of <parameter>=<value>")
            sys.exit(1)
        if not tokens[0] in parameters and tokens[0] != "configuration" and tokens[0] != "gpus_number" and tokens[0] != "n" and tokens[0] != "gpus_instances":
            logging.error("parameter %s is not present in the source configuration", tokens[0])
            sys.exit(1)
        parameters[tokens[0]] = tokens[1]
    return parameters

def compute_configuration_name(cl_parameters):
    parameters = compute_parameters(cl_parameters)
    if "gpus_number" in parameters:
        gpus_number = "_gpus_number_" + parameters["gpus_number"]
    else:
        gpus_number = ""
    network_type = parameters["network_type"]
    if network_type == "resnet":
        if "n" in parameters:
            deep = parameters["n"]
        else:
            deep = "5"
        network_type = "resnet_" + deep + "_deep"
    if "only_load" in parameters and parameters["only_load"] == "True":
        only_load = "_only_load"
    else:
        only_load = ""
    configuration_name = network_type + "_cl_" + parameters["num_classes"] + "_im_" + parameters["images_per_class"] + "_ep_" + parameters["epochs_number"] + "_bs_" + parameters["batch_size"] + "_mo_" + parameters["momentum"] + "_j_" + parameters["j"] + gpus_number + only_load
    return configuration_name

def load_xml_configuration(xml_configuration_file):
    #The absolute path of the current file
    abs_script = os.path.realpath(__file__)

    #The root directory of the script
    abs_root = os.path.dirname(abs_script)

    #The absolute path of the configuration directory
    confs_dir = os.path.join(abs_root, "pytorch", "confs")
    logging.info("conf directory is %s", confs_dir)

    #Check the confs_dir exists
    if not os.path.exists(confs_dir):
        logging.error("Conf directory %s does not exist", confs_dir)
        sys.exit(1)

    #Check if xml file of the conf exist
    xml_file_name = os.path.join(confs_dir, xml_configuration_file)
    if not os.path.exists(xml_file_name):
        logging.error("XML file %s not found", xml_file_name)
        sys.exit(1)


    #Load XML file
    with open(xml_file_name) as xml_file:
        doc = xmltodict.parse(xml_file.read(), force_list={'input_class'})
    return doc


def collect_data(repetition_path, gpu_type, gpu_number, debug):
    try:
        #The iterations fractions
        iteration_fractions = [0.25, 0.50, 0.75]

        #The number of initial iterations to be skipped
        skipped_initial_iterations = 20

        #The absolute path of the current file
        abs_script = os.path.realpath(__file__)

        #The root directory of the script
        abs_root = os.path.dirname(abs_script)

        #Skip experiment for boot gpu
        if repetition_path.find("alexnet_cl_3_im_9999_ep_1_bs_256_mo_0.9") != -1 or repetition_path.find("alexnet_cl_2_im_9999_ep_1_bs_1024_mo_0.9") != -1:
            return
        csv_file_name = "pytorch.csv"
        if os.path.exists(csv_file_name):
            csv_file = open(csv_file_name, "a")
        else:
            csv_file = open(csv_file_name, "w")
            column_labels = [
                "starting timestamp",
                "starting time",
                "pytorch version",
                "system_UUID",
                "mac_address",
                "vm_instance",
                "GPU type",
                "GFlops",
                "disk speed",
                "GPU number",
                "Network Type",
                "Network depth",
                "classes",
                "batch size",
                "Iterations Fraction",
                "Real Iterations Number",
                "Computed Iterations Number",
                "Skipped Iterations Number",
                "epochs",
                "CPU threads",
                "profile",
                "cpus usage",
                "Average CPU usage",
                "gpus usage",
                "Average GPU Usage",
                "usage ratio",
                "only_load",
                "overlapped",
                "data time",
                "training time",
                "test time",
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
        if not os.path.exists("pytorch_csvs"):
            os.mkdir("pytorch_csvs")

        execution_stdout_filename = os.path.join(repetition_path, "execution_stdout")
        execution_stderr_filename = os.path.join(repetition_path, "execution_stderr")
        hw_configuration_filename = os.path.join(repetition_path, "hw_configuration")
        overlapped_filename = os.path.join(repetition_path, "overlapped")
        if os.stat(execution_stdout_filename).st_size == 0:
            return
        if not os.path.exists(hw_configuration_filename):
            return
        if os.path.exists(overlapped_filename):
            overlapped = "1"
        else:
            overlapped = "0"
        execution_stdout = open(execution_stdout_filename)
        execution_stderr = open(execution_stderr_filename)
        hw_configuration = open(hw_configuration_filename)
        #Checking stderr for error
        error_found = False
        for line in execution_stderr:
            if line.find("THCudaCheck FAIL") != -1:
                error_found = True
                break
            if line.find("RuntimeError: CUDA out of memory") != -1:
                error_found = True
                break
            if line.find("RuntimeError: CUDA error: out of memory") != -1:
                error_found = True
                break
            if line.find("RuntimeError: cublas runtime error") != -1:
                error_found = True
                break
            if line.find("main.py: error:") != -1:
                error_found = True
                break
            if line.find("ValueError: invalid literal for int() with base 10") != -1:
                error_found = True
                break
            if line.find("No space left on device") != -1:
                error_found = True
                break
            if line.find("Permission denied") != -1:
                error_found = True
                break
        if error_found:
            csv_file.close()
            return
        training_files_number = ""
        starting_timestamp = ""
        training_time = 0.0
        data_time = 0.0
        test_time = 0.0
        current_iteration = 0
        initial_training_time = 0.0
        initial_data_time = 0.0
        for line in execution_stdout:
            if line.startswith("Overall training time is"):
                overall_execution_time = line.split()[4]
                times_vector = re.split(r'\(|=|>|\)', line)
                starting_timestamp = int((datetime.datetime.strptime(times_vector[1].strip(), "%a %b %d %H:%M:%S %Y") - datetime.datetime(1970, 1, 1)).total_seconds())
                starting_time = (datetime.datetime.strptime(times_vector[1].strip(), "%a %b %d %H:%M:%S %Y")).strftime("%Y-%m-%d %H:%M:%S")
                ending_time = (datetime.datetime.strptime(times_vector[4].strip(), "%a %b %d %H:%M:%S %Y")).strftime("%Y-%m-%d %H:%M:%S")
            if line.startswith("Number of training files is"):
                training_files_number = line.split()[5]
            if line.startswith("Epoch: "):
                iteration_data_time = line.split('\t')[2].split()[1]
                iteration_training_time = float(line.split('\t')[1].split()[1]) - float(iteration_data_time)
                training_time = training_time + float(iteration_training_time)
                data_time = data_time + float(iteration_data_time)
                if current_iteration < skipped_initial_iterations:
                    initial_training_time = initial_training_time + float(iteration_training_time)
                    initial_data_time = initial_data_time + float(iteration_data_time)
                current_iteration = current_iteration + 1
            if line.startswith("Test: "):
                iteration_time = line.split('\t')[1].split()[1]
                test_time = test_time + float(iteration_time)

        xml_configuration_file_name = os.path.join(repetition_path, "configuration.xml")
        if not os.path.exists(xml_configuration_file_name):
            csv_file.close()
            return
        if starting_timestamp == "":
            csv_file.close()
            return
        xml_configuration = xmltodict.parse(open(xml_configuration_file_name).read(), force_list={'input_class'})["pytorch_configuration"]
        configuration_gpus_number = xml_configuration.get("gpus_number")
        if configuration_gpus_number != None:
            gpu_number = int(configuration_gpus_number)


        #Prepering csv file with details about iteration time executions
        current_epoch = 0
        repetition_number = os.path.basename(repetition_path)
        configuration_path = os.path.basename(os.path.dirname(os.path.dirname(repetition_path)))
        experiment_path = os.path.basename(os.path.dirname(repetition_path))
        end_information = False
        iterations_number = 0
        iteration_file_name = os.path.join("pytorch_csvs", "iterations_" + gpu_type.replace(" ", "-") + "_" + str(gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(starting_timestamp) + ".csv")
        if not os.path.exists(iteration_file_name):
            iteration_file = open(iteration_file_name, "w")
            iteration_file.write("Epoch,Phase,Iteration,DataTime,TrainingTime,Testtime,End\n")
            for line in open(execution_stdout_filename):
                cleaned_line = line
                while cleaned_line.find("[ ") != -1:
                    cleaned_line = cleaned_line.replace("[ ", "[")
                if cleaned_line.startswith("Epoch: "):
                    epoch_iteration = cleaned_line.split('\t')[0].split()[1]
                    split = re.split(r"\[|\]|/", epoch_iteration)
                    epoch = split[1]
                    if int(epoch) > current_epoch:
                        current_epoch = int(epoch)
                    iteration = split[3]
                    if iteration == "":
                        logging.error("unexpected pattern: %s", line)
                        logging.error("%s", str(split))
                        sys.exit(1)
                    iteration_data_time = cleaned_line.split('\t')[2].split()[1]
                    iteration_training_time = float(cleaned_line.split('\t')[1].split()[1]) - float(iteration_data_time)
                    if cleaned_line.find("End") != -1:
                        end = cleaned_line.split('\t')[6].split()[1]
                        end_information = True
                    else:
                        end = "NaN"
                    iteration_file.write(epoch + ",Training," + iteration + "," + iteration_data_time + "," + str(iteration_training_time) + ",0.0," + end + "\n")
                    iterations_number = iterations_number + 1
                if cleaned_line.startswith("Test: "):
                    iteration = re.split(r"\[|\]|/", cleaned_line.split('\t')[0].split()[1])[1]
                    iteration_time = cleaned_line.split('\t')[1].split()[1]
                    if cleaned_line.find("End") != -1:
                        end = cleaned_line.split('\t')[6].split()[1]
                        end_information = True
                    else:
                        end = "NaN"
                    iteration_file.write(str(current_epoch) + ",Testing," + str(iteration) + ",0.0,0.0," + iteration_time + "," + end + "\n")
            iteration_file.close()
        else:
            for line in open(iteration_file_name):
                if line.find("Training") != -1:
                    iterations_number = iterations_number + 1
        if os.path.exists(iteration_file_name) and not os.path.exists(iteration_file_name.replace(".csv", ".pdf")):
            create_graph_command = os.path.join(abs_root, "pytorch", "generate_iteration_graph.py") + " " + iteration_file_name + " -o " + iteration_file_name.replace(".csv", ".pdf") + " -s" + repetition_path
            logging.debug("Executing %s", create_graph_command)
            return_value = subprocess.call(create_graph_command, shell=True, executable="/bin/bash")
            if return_value:
                logging.error("Error in execution of %s", create_graph_command)
                sys.exit(1)

        #Preparing csv file with cpu and gpu utilization
        profile_cpu_output_filename = os.path.join(repetition_path, "profile_cpu_output")
        profile_gpu_output_filename = os.path.join(repetition_path, "profile_gpu_output")
        profile_file_name_cpu = os.path.join("pytorch_csvs", "profile_cpu_" + gpu_type.replace(" ", "-") + "_" + str(gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(starting_timestamp) + ".csv")
        profile_file_name_sum_cpu = os.path.join("pytorch_csvs", "profile_sum_cpu_" + gpu_type.replace(" ", "-") + "_" + str(gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(starting_timestamp) + ".csv")
        if os.path.exists(profile_cpu_output_filename) and (not os.path.exists(profile_file_name_cpu) or not os.path.exists(profile_file_name_sum_cpu)):
            #The collected data
            cpu_data = {}
            cpu_sum_data = {}
            maximum_cpu_number = 0
            current_timestamp = ""
            previous_timestamp = ""
            #Analyzing profile_cpu_output
            for line in open(profile_cpu_output_filename, "r"):
                #New entry
                if line.find("%cpu %MEM ARGS") != -1:
                    previous_timestamp = current_timestamp
                    #Old pattern
                    if line.startswith("%cpu %MEM ARGS"):
                        split = line.split()
                        if len(split) == 5:
                            read_timestamp = split[3] + " " + split[4][0:7]
                            current_timestamp_datetime = datetime.datetime.strptime(read_timestamp, "%Y-%m-%d %X")
                            current_timestamp_readable = current_timestamp_datetime.strftime("%Y-%m-%d %x")
                            current_timestamp = str(int(current_timestamp_datetime.timestamp()))
                        else:
                            read_timestamp = split[4] + " " + split[5] + " " + split[6] + " " + split[8]
                            current_timestamp_datetime = datetime.datetime.strptime(read_timestamp, "%b %d %H:%M:%S %Y")
                            current_timestamp_readable = current_timestamp_datetime.strftime("%Y-%m-%d %x")
                            current_timestamp = str(int(current_timestamp_datetime.timestamp()))
                    #New pattern
                    else:
                        split = line.replace("\\n%cpu %MEM ARGS", "").split()
                        current_timestamp_readable = split[4] + " " + split[5]
                        current_timestamp = split[1]
                    logging.debug("Found timestamp %s (%s(", current_timestamp, current_timestamp_readable)
                    #Workaround: some profile logs contains data of next repetition
                    if current_timestamp_readable > ending_time:
                        break
                    current_cpu_number = 0
                    cpu_data[current_timestamp] = {}
                    cpu_data[current_timestamp]["readable_timestamp"] = current_timestamp_readable
                elif line.find("apps/pytorch/main.py") != -1:
                    split = line.split()
                    cpu_usage = split[0]
                    cpu_data[current_timestamp]["cpu" + str(current_cpu_number)] = cpu_usage
                    if not "cpu" + str(current_cpu_number) in cpu_sum_data:
                        cpu_sum_data["cpu" + str(current_cpu_number)] = 0.0
                    if previous_timestamp != "":
                        cpu_sum_data["cpu" + str(current_cpu_number)] = cpu_sum_data["cpu" + str(current_cpu_number)] + (float(cpu_usage) * (float(current_timestamp) - float(previous_timestamp))/1000000000)
                    current_cpu_number = current_cpu_number + 1
                    if current_cpu_number > maximum_cpu_number:
                        maximum_cpu_number = current_cpu_number
            #Writing results in csv file
            profile_file = open(profile_file_name_cpu, "w")
            profile_file.write("timestamp,readable timestamp")
            for cpu_number in range(0, maximum_cpu_number):
                profile_file.write(",cpu" + str(cpu_number))
            profile_file.write("\n")
            for timestamp in sorted(cpu_data):
                profile_file.write(timestamp)
                current_data = cpu_data[timestamp]
                profile_file.write("," + current_data["readable_timestamp"])
                for cpu_number in range(0, maximum_cpu_number):
                    if "cpu" + str(cpu_number) in current_data:
                        profile_file.write("," + current_data["cpu" + str(cpu_number)])
                    else:
                        profile_file.write(",0")
                profile_file.write("\n")
            profile_file.close()
            #Writing sum results in csv file
            profile_sum_file = open(profile_file_name_sum_cpu, "w")
            for cpu_number in range(0, maximum_cpu_number):
                if cpu_number != 0:
                    profile_sum_file.write(",")
                profile_sum_file.write("cpu" + str(cpu_number))
            profile_sum_file.write("\n")
            for cpu_number in range(0, maximum_cpu_number):
                if cpu_number != 0:
                    profile_sum_file.write(",")
                profile_sum_file.write(str(cpu_sum_data["cpu" + str(cpu_number)]))
            profile_sum_file.write("\n")
            profile_sum_file.close()
        profile_file_name_gpu = os.path.join("pytorch_csvs", "profile_gpu_" + gpu_type.replace(" ", "-") + "_" + str(gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(starting_timestamp) + ".csv")
        profile_file_name_sum_gpu = os.path.join("pytorch_csvs", "profile_sum_gpu_" + gpu_type.replace(" ", "-") + "_" + str(gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(starting_timestamp) + ".csv")

        if os.path.exists(profile_gpu_output_filename) and (not os.path.exists(profile_file_name_gpu) or not os.path.exists(profile_file_name_sum_gpu)):
            #The collected data
            gpu_data = {}
            gpu_sum_data = {}
            #The maximum number of gpu used
            maximum_gpu_number = 0
            previous_timestamp = ""
            current_timestamp = ""
            #Analyzing profile_gpu_output
            for line in open(profile_gpu_output_filename, "r"):
                logging.debug("Read %s", line)
                split = line.split()
                #New pattern
                if line.find("Timestamp") != -1:
                    previous_timestamp = current_timestamp
                    current_timestamp = split[1]
                    current_timestamp_readable = split[4] + " " + split[5]
                    gpu_data[current_timestamp] = {}
                    gpu_data[current_timestamp]["readable_timestamp"] = current_timestamp_readable
                    current_gpu_number = 0
                    logging.debug("Found timestamp %s", current_timestamp)
                    #Workaround: some profile logs contains data of next repetition
                    if current_timestamp_readable > ending_time:
                        break
                    continue
                #Old pattern
                if line.find("%") == -1 and line.find("utilization") == -1:
                    previous_timestamp = current_timestamp
                    split = line.split()
                    if len(split) == 2:
                        read_timestamp = split[0] + " " + split[1][0:7]
                        current_timestamp_datetime = datetime.datetime.strptime(read_timestamp, "%Y-%m-%d %X")
                        current_timestamp_readable = current_timestamp_datetime.strftime("%Y-%m-%d %x")
                        current_timestamp = str(int(current_timestamp_datetime.timestamp()))
                    else:
                        read_timestamp = split[1] + " " + split[2] + " " + split[3] + " " + split[5]
                        current_timestamp_datetime = datetime.datetime.strptime(read_timestamp, "%b %d %H:%M:%S %Y")
                        current_timestamp_readable = current_timestamp_datetime.strftime("%Y-%m-%d %x")
                        current_timestamp = str(int(current_timestamp_datetime.timestamp()))
                    gpu_data[current_timestamp] = {}
                    gpu_data[current_timestamp]["readable_timestamp"] = current_timestamp_readable
                    current_gpu_number = 0
                    logging.debug("Found timestamp (Old pattern) %s", current_timestamp)
                    #Workaround: some profile logs contains data of next repetition
                    if current_timestamp_readable > ending_time:
                        break
                    continue
                if line.find("utilization") != -1:
                    continue
                #The line actually stores gpu usage information
                gpu_usage = split[0]
                memory_usage = split[2]
                if not current_timestamp in gpu_data:
                    gpu_data[current_timestamp] = {}
                gpu_data[current_timestamp]["gpu" + str(current_gpu_number)] = gpu_usage
                if not "gpu" + str(current_gpu_number) in gpu_sum_data:
                    gpu_sum_data["gpu" + str(current_gpu_number)] = 0.0
                if previous_timestamp != "":
                    gpu_sum_data["gpu" + str(current_gpu_number)] = gpu_sum_data["gpu" + str(current_gpu_number)] + (float(gpu_usage) * (float(current_timestamp) - float(previous_timestamp))/1000000000)
                gpu_data[current_timestamp]["gpu" + str(current_gpu_number) + "memory"] = memory_usage
                current_gpu_number = current_gpu_number + 1
                logging.debug("Found gpu utilization. Number of gpu updated to %d", current_gpu_number)
                if current_gpu_number > maximum_gpu_number:
                    maximum_gpu_number = current_gpu_number
            #Writing results in csv file
            profile_file = open(profile_file_name_gpu, "w")
            profile_file.write("timestamp,readable timestamp")
            for local_gpu_number in range(0, maximum_gpu_number):
                profile_file.write(",gpu" + str(local_gpu_number))
            for local_gpu_number in range(0, maximum_gpu_number):
                profile_file.write(",gpu" + str(local_gpu_number) + "memory")
            profile_file.write("\n")
            for timestamp in sorted(gpu_data):
                profile_file.write(timestamp)
                current_data = gpu_data[timestamp]
                profile_file.write("," + current_data["readable_timestamp"])
                for local_gpu_number in range(0, maximum_gpu_number):
                    if "gpu" + str(local_gpu_number) in current_data:
                        profile_file.write("," + current_data["gpu" + str(local_gpu_number)])
                    else:
                        profile_file.write(",0")
                for local_gpu_number in range(0, maximum_gpu_number):
                    if "gpu" + str(local_gpu_number) + "memory" in current_data:
                        profile_file.write("," + current_data["gpu" + str(local_gpu_number) + "memory"])
                    else:
                        profile_file.write(",0")
                profile_file.write("\n")
            profile_file.close()
            #Writing sum results in csv file
            profile_sum_file = open(profile_file_name_sum_gpu, "w")
            for local_gpu_number in range(0, maximum_gpu_number):
                if local_gpu_number != 0:
                    profile_sum_file.write(",")
                profile_sum_file.write("gpu" + str(local_gpu_number))
            profile_sum_file.write("\n")
            for local_gpu_number in range(0, maximum_gpu_number):
                if local_gpu_number != 0:
                    profile_sum_file.write(",")
                profile_sum_file.write(str(gpu_sum_data["gpu" + str(local_gpu_number)]))
            profile_sum_file.write("\n")
            profile_sum_file.close()
        profile_file_name = os.path.join("pytorch_csvs", "profile_" + gpu_type.replace(" ", "-") + "_" + str(gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(starting_timestamp) + ".pdf")


        if os.path.exists(profile_file_name_cpu) and os.path.exists(profile_file_name_gpu) and not os.path.exists(profile_file_name):
            create_graph_command = os.path.join(abs_root, "pytorch", "generate_profile_graph.py") + " -c" + profile_file_name_cpu + " -g" + profile_file_name_gpu + " -o" + profile_file_name + " -s" + repetition_path + " -t" + overall_execution_time
            if end_information:
                create_graph_command = create_graph_command + " -i" + iteration_file_name
            if debug:
                create_graph_command = create_graph_command + " -d"
            logging.debug("Executing %s", create_graph_command)
            return_value = subprocess.call(create_graph_command, shell=True, executable="/bin/bash")
            if return_value != 0:
                logging.error("Error in analyzing result of %s", repetition_path)
                sys.exit(-1)

        network_type = xml_configuration["network_type"]
        if network_type == "resnet":
            n_value = xml_configuration["n"]
        else:
            n_value = "NaN"
        num_classes = xml_configuration["num_classes"]
        batch_size = xml_configuration["batch_size"]
        epochs_number = xml_configuration["epochs_number"]
        computed_iterations_number = xml_configuration.get("iteration_number")
        if computed_iterations_number is None:
            computed_iterations_number = str(int(training_files_number) * int(epochs_number) / int(batch_size))
        pytorch_version = xml_configuration.get("pytorch_version")
        if pytorch_version is None:
            pytorch_version = "unknown"
        j = xml_configuration.get("j")
        if j is None:
            j = "4"

        if xml_configuration.get("only_load") and xml_configuration.get("only_load") == "True":
            only_load = "1"
        else:
            only_load = "0"

        #Retrieving machine information
        #Add host_scripts to the directories for python packages search
        host_scripts_path = os.path.join(abs_root, "..", "host_scripts")
        sys.path.append(host_scripts_path)
        collect_data_package = __import__("collect_data")

        mac_address = ""
        system_uuid = ""
        machine_name = ""

        #Retrieving information
        for line in hw_configuration:
            if line.find("serial: ") != -1:
                mac_address = line.split()[1]
            if line.startswith("Linux "):
                machine_name = line.split()[1]
        if xml_configuration.get("system_UUID"):
            system_uuid = xml_configuration.get("system_UUID")

        machine_information = collect_data_package.get_machine_information(mac_address, machine_name, system_uuid)
        mac_address = machine_information["mac_address"]
        system_uuid = machine_information["system_uuid"]
        machine_name = machine_information["machine_name"]
        gflops = machine_information["gflops"]
        disk_speed = machine_information["disk_speed"]

        profile = xml_configuration.get("profile")
        if profile is None or not os.path.exists(profile_file_name_sum_cpu) or not os.path.exists(profile_file_name_sum_gpu):
            profile = "NaN"
            cpu_usage = "NaN"
            average_cpu_usage = "NaN"
            gpu_usage = "NaN"
            average_gpu_usage = "NaN"
            usage_ratio = "NaN"
        else:
            cpu_usage = 0.0
            gpu_usage = 0.0
            cpu_row_1 = linecache.getline(profile_file_name_sum_cpu, 2)
            split = cpu_row_1.split(",")
            for token in split:
                cpu_usage = cpu_usage + float(token.replace("\n", ""))
            average_cpu_usage = cpu_usage/float(overall_execution_time)
            gpu_row_1 = linecache.getline(profile_file_name_sum_gpu, 2)
            split = gpu_row_1.split(",")
            for token in split:
                gpu_usage = gpu_usage + float(token.replace("\n", ""))
            average_gpu_usage = gpu_usage/float(overall_execution_time)
            usage_ratio = cpu_usage/gpu_usage

        missing_time = float(overall_execution_time) - data_time - training_time - test_time

        #Computing the values of fractions of this experiment
        iteration_number_fractions = {}
        data_time_fractions = {}
        training_time_fractions = {}
        if iteration_fractions:
            current_iteration = 0
            current_aggregated_data_time = 0.0
            current_aggregated_training_time = 0.0
            iteration_fractions_index = 0
            current_iterations_fraction_number = int(round(iterations_number * iteration_fractions[iteration_fractions_index]))
            iteration_number_fractions[iteration_fractions[iteration_fractions_index]] = current_iterations_fraction_number
            for line in open(iteration_file_name):
                if line.find("Training,") != -1:
                    current_iteration = current_iteration + 1
                    split = line.split(",")
                    current_date_time = float(split[3])
                    current_training_time = float(split[4])
                    current_aggregated_data_time = current_aggregated_data_time + current_date_time
                    current_aggregated_training_time = current_aggregated_training_time + current_training_time
                    if current_iteration == current_iterations_fraction_number:
                        data_time_fractions[iteration_fractions[iteration_fractions_index]] = current_aggregated_data_time
                        training_time_fractions[iteration_fractions[iteration_fractions_index]] = current_aggregated_training_time
                        iteration_fractions_index = iteration_fractions_index + 1
                        if iteration_fractions_index < len(iteration_fractions):
                            current_iterations_fraction_number = int(round(iterations_number * iteration_fractions[iteration_fractions_index]))
                            iteration_number_fractions[iteration_fractions[iteration_fractions_index]] = current_iterations_fraction_number
                        else:
                            break
        for iteration_fraction in iteration_fractions:
            iteration_number_fraction = iteration_number_fractions[iteration_fraction]
            epochs_number_fraction = str(float(epochs_number) * iteration_fraction)
            data_time_fraction = data_time_fractions[iteration_fraction]
            training_time_fraction = training_time_fractions[iteration_fraction]
            csv_file.write(str(starting_timestamp) + "," + starting_time + "," + pytorch_version + "," + system_uuid + "," + mac_address + "," + machine_name + "," + gpu_type + "," + gflops + "," + disk_speed + "," + str(gpu_number) + "," + network_type + "," + n_value + "," + num_classes + "," + batch_size + "," + str(iteration_fraction) + "," + str(iteration_number_fraction) + "," + str(float(computed_iterations_number)*iteration_fraction) + ",0," + epochs_number_fraction + "," + j + "," + profile + ",NaN,NaN,NaN,NaN,NaN," + only_load + "," + overlapped + "," + str(data_time_fraction) + "," + str(training_time_fraction) + ",NaN,NaN,NaN," + repetition_number + "," + repetition_path + "\n")
            if iteration_number_fraction > skipped_initial_iterations:
                csv_file.write(str(starting_timestamp) + "," + starting_time + "," + pytorch_version + "," + system_uuid + "," + mac_address + "," + machine_name + "," + gpu_type + "," + gflops + "," + disk_speed + "," + str(gpu_number) + "," + network_type + "," + n_value + "," + num_classes + "," + batch_size + "," + str(iteration_fraction) + "," + str(iteration_number_fraction - skipped_initial_iterations) + "," + str(float(computed_iterations_number)*iteration_fraction - skipped_initial_iterations) + "," + str(skipped_initial_iterations) + "," + epochs_number_fraction + "," + j + "," + profile + ",NaN,NaN,NaN,NaN,NaN," + only_load + "," + overlapped + "," + str(data_time_fraction - initial_data_time) + "," + str(training_time_fraction - initial_training_time) + ",NaN,NaN,NaN," + repetition_number + "," + repetition_path + "\n")


        #Writing full experiment data
        csv_file.write(str(starting_timestamp) + "," + starting_time + "," + pytorch_version + "," + system_uuid + "," + mac_address + "," + machine_name + "," + gpu_type + "," + gflops + "," + disk_speed + "," + str(gpu_number) + "," + network_type + "," + n_value + "," + num_classes + "," + batch_size + ",1.0," + str(iterations_number) + "," + computed_iterations_number + ",0," + epochs_number + "," + j + "," + profile + "," + str(cpu_usage) + "," + str(average_cpu_usage) + "," + str(gpu_usage) + "," + str(average_gpu_usage) + "," + str(usage_ratio) + "," + only_load + "," + overlapped + "," + str(data_time) + "," + str(training_time) + "," + str(test_time) + "," + overall_execution_time + "," + str(missing_time) + "," + repetition_number + "," + repetition_path + "\n")
        if iterations_number > skipped_initial_iterations:
            csv_file.write(str(starting_timestamp) + "," + starting_time + "," + pytorch_version + "," + system_uuid + "," + mac_address + "," + machine_name + "," + gpu_type + "," + gflops + "," + disk_speed + "," + str(gpu_number) + "," + network_type + "," + n_value + "," + num_classes + "," + batch_size + ",1.0," + str(iterations_number-skipped_initial_iterations) + "," + str(float(computed_iterations_number)-skipped_initial_iterations) + "," + str(skipped_initial_iterations) + "," + epochs_number + "," + j + "," + profile + "," + str(cpu_usage) + "," + str(average_cpu_usage) + "," + str(gpu_usage) + "," + str(average_gpu_usage) + "," + str(usage_ratio) + "," + only_load + "," + overlapped + "," + str(data_time - initial_data_time) + "," + str(training_time - initial_training_time) + "," + str(test_time) + "," + overall_execution_time + "," + str(missing_time) + "," + repetition_number + "," + repetition_path + "\n")
        csv_file.close()

    except:
        logging.error("Error in analyzing result of %s", repetition_path)
        raise

def main():
    #The absolute path of the current file
    abs_script = os.path.realpath(__file__)

    #The root directory of the script
    abs_root = os.path.dirname(abs_script)

    #The return value of the command
    return_value = 0

    #Parsing input arguments
    parser = argparse.ArgumentParser(description="Train an alexnet network by means pytorch")
    parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    parser.add_argument('-p', "--parameters", help="Parameters to be overwritten", required=True)
    parser.add_argument("--no-clean", help="Do not delete generated files", default=False)
    args = parser.parse_args()

    #Initializing logger
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #Root tree node
    root = compute_parameters(args.parameters)

    #Checking input classes
    input_classes = root.get("input_classes")
    if input_classes is None:
        logging.error("inputs tag not found in %s.xml", args.configuration)
        sys.exit(1)

    local_path = input_classes.get("local_path")
    if local_path is None:
        logging.error("local_path is not specified in input_classes")
        sys.exit(1)

    #Expand ~ when present
    if local_path.find("~") != -1:
        local_path = os.path.expanduser(local_path)

    #If it does not exist, create local path root
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    #If it does not exist, create train directory
    train_path = os.path.join(local_path, "train")
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    #If it does not exist, create val directory
    val_path = os.path.join(local_path, "val")
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    #Collect information about remote position of inputs
    remote_location = input_classes.get("remote_location")
    if remote_location is None:
        logging.error("remote_position is not specified in input classes")
        sys.exit(1)
    remote_user = remote_location.get("user")
    if remote_user is None:
        logging.error("user not set in remote_position of input classes")
        sys.exit(1)
    remote_key = remote_location.get("key")
    if remote_key is None:
        logging.error("key not set in remote position of input classes")
        sys.exit(1)
    remote_host = remote_location.get("host")
    if remote_host is None:
        logging.error("host not set in remote position of input classes")
        sys.exit(1)
    remote_path = remote_location.get("path")
    if remote_path is None:
        logging.error("path not set in remote position of input classes")
        sys.exit(1)

    #Get number of classes
    num_classes = root.get("num_classes")
    if num_classes is None:
        logging.error("Number of classes not set in xml file")
        sys.exit(1)

    #Copy input data to local machine if necessary
    if len(input_classes.get("input_class")) < 2:
        logging.error("input classes not set")
        sys.exit(1)
    added_num_classes = 0
    for input_class in input_classes.get("input_class"):
        if added_num_classes == int(num_classes):
            break
        added_num_classes = added_num_classes + 1
        logging.info("Found input class %s", input_class)
        #Check if the input class files are already available locally
        if os.path.exists(os.path.join(train_path, input_class)) and os.path.exists(os.path.join(val_path, input_class)):
            continue
        rsync_command = "rsync -a -e \"ssh -i " + remote_key + " -o StrictHostKeyChecking=no\" " + remote_user + "@" + remote_host + ":" + os.path.join(remote_path, "train", input_class) + " " + train_path
        logging.info("rsync command is %s", rsync_command)
        subprocess.call(rsync_command, shell=True, executable="/bin/bash")
        rsync_command = "rsync -a -e \"ssh -i " + remote_key + " -o StrictHostKeyChecking=no\" " + remote_user + "@" + remote_host + ":" + os.path.join(remote_path, "val", input_class) + " " + val_path
        logging.info("rsync command is %s", rsync_command)
        subprocess.call(rsync_command, shell=True, executable="/bin/bash")

    #Create temporary directory with experiment input
    temporary_directory = tempfile.mkdtemp()
    temporary_train = os.path.join(temporary_directory, "train")
    os.makedirs(temporary_train)
    temporary_val = os.path.join(temporary_directory, "val")
    os.makedirs(temporary_val)
    added_num_classes = 0
    for input_class in input_classes.get("input_class"):
        os.symlink(os.path.join(train_path, input_class), os.path.join(temporary_train, input_class))
        os.symlink(os.path.join(val_path, input_class), os.path.join(temporary_val, input_class))
        added_num_classes = added_num_classes + 1
        if added_num_classes == int(num_classes):
            break

    #Computing number of files
    training_files_number = 0
    validation_files = 0
    added_num_classes = 0
    for input_class in input_classes.get("input_class"):
        training_files_number = training_files_number + len(os.listdir(os.path.join(temporary_train, input_class)))
        validation_files = validation_files + len(os.listdir(os.path.join(temporary_val, input_class)))
        added_num_classes = added_num_classes + 1
        if added_num_classes == int(num_classes):
            break

    #Get network type
    network_type = root.get("network_type")
    if network_type is None:
        logging.error("Network type not set in xml file")
        sys.exit(1)
    #If network is resnet build it on the basis of n
    if network_type == "resnet":
        n_value = root.get("n")
        if n_value != None:
            deep = n_value
        else:
            deep = 5
        network_type = "resnet_" + deep + "_deep"

    #Get number of epochs
    epochs_number = root.get("epochs_number")
    if epochs_number is None:
        logging.error("Epoch number not set in xml file")
        sys.exit(1)

    #Get batch size
    batch_size = root.get("batch_size")
    if batch_size is None:
        logging.error("Batch size not set in xml file")
        sys.exit(1)

    #Get momentum
    momentum = root.get("momentum")
    if momentum is None:
        logging.error("Momentum not set in xml file")
        sys.exit(1)

    #Get j
    j = root.get("j")
    if j is None:
        logging.error("J not set in xml file")
        sys.exit(1)

    only_load = root.get("only_load")
    if only_load and only_load == "True":
        only_load = " --only-load"
    else:
        only_load = ""

    #If the number of gpus is specified, uses it, otherwise leave default (all gpus will be used)
    gpus_number = root.get("gpus_number")
    gpus_instances = root.get("gpus_instances")
    #If the instances of gpus have been specified
    if gpus_instances is not None:
        #Check that gpus instances and gpus number are consistent
        if len(gpus_instances.split("_")) != int(gpus_number):
            logging.error("gpus_number is " + gpus_number + " while gpus_instances are " + gpus_instances)
            sys.exit(1)
        export_gpus_command = "CUDA_VISIBLE_DEVICES=" + gpus_instances.replace("_", ",") + " "
    elif gpus_number is None:
        export_gpus_command = ""
    else:
        export_gpus_command = "CUDA_VISIBLE_DEVICES=" + ",".join(str(gpu) for gpu in list(range(0, int(gpus_number)))) + " "

    #Computing number of iterations
    root["iteration_number"] = (int(training_files_number)/int(batch_size)) * int(epochs_number)

    #Adding pytorch version
    import torch
    root["pytorch_version"] = torch.__version__

    #Adding system UUID
    if not os.path.exists("/etc/machine-id"):
        logging.warning("/etc/machine-id does not exists")
    else:
        uuid_line = open("/etc/machine-id", "r").readline()
        if len(uuid_line.split()) != 2:
            logging.error("Error in loading uuid: %s", str(uuid_line.split()))
            sys.exit(1)
        uuid = uuid_line.split()[1]

        root["system_UUID"] = uuid

    #Dump configuration in xml
    dicttoxml.set_debug(False)
    generated_xml = parseString(dicttoxml.dicttoxml(root, custom_root="pytorch_configuration", attr_type=False)).toprettyxml(indent="   ")
    generated_xml_file = open("configuration.xml", "w")
    generated_xml_file.write(generated_xml)
    generated_xml_file.close()

    #Perform the acutal nn training
    imagenet_script = os.path.join(abs_root, "pytorch", "main.py")
    imagenet_command = export_gpus_command + "python3 " + imagenet_script + " --print-freq 1 --arch " + network_type + " --epochs " + epochs_number + " --batch-size " + batch_size + " --momentum " + momentum + " -j" + j + only_load + " "  + temporary_directory
    logging.info("imagenet command is %s", imagenet_command)
    starting_time = time.time()
    return_value = return_value or subprocess.call(imagenet_command, shell=True, executable="/bin/bash")
    ending_time = time.time()
    if not args.no_clean:
        if os.path.exists("checkpoint.pth.tar"):
            os.remove("checkpoint.pth.tar")
        if os.path.exists("model_best.pth.tar"):
            os.remove("model_best.pth.tar")

    #Dump information about experiment
    print("Pytorch nn training")
    print("Network type is " + network_type)
    print("Number of classes is " + str(num_classes))
    print("Number of training files is " + str(training_files_number))
    print("Overall training time is " + str(ending_time - starting_time) + " seconds (" + time.ctime(starting_time) + " ==> " + time.ctime(ending_time) + ")")
    print("Configuration file is:")
    print(generated_xml)

    sys.exit(return_value)

if __name__ == "__main__":
    main()
