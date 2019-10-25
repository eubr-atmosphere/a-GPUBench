#!/usr/bin/env python3
"""
Copyright 2018 Marco Speziali

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
import shutil
import subprocess

import sys
import tempfile
from typing import IO
from xml.dom.minidom import parseString

import time

import dicttoxml
import xmltodict

# The absolute path of the current file
abs_script = os.path.realpath(__file__)

# The root directory of the script
abs_root = os.path.dirname(abs_script)


def compute_configuration_name(cl_parameters):
    parameters = compute_configuration(cl_parameters)
    configuration_name = parameters["network_type"] + "_cl_" + parameters["num_classes"] + "_im_" + parameters[
        "images_per_class"] + "_ep_" + parameters["epochs_number"] + "_bs_" + parameters["batch_size"] + "_mo_" + \
                         parameters["momentum"]
    return configuration_name


def parse_arguments() -> {}:
    """
    Parses the arguments and validates them.
    :rtype: {}
    :return: The parsed arguments if correctly validated.
    """
    # Parsing input arguments
    parser = argparse.ArgumentParser(description="Trains an alexnet network by means of tf")
    parser.add_argument(
        '-d', "--debug",
        help="Enable debug messages",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        '-p', "--parameters",
        help="Parameters to be overwritten",
        required=False
    )
    parser.add_argument(
        "--no-clean",
        help="Do not delete generated files",
        default=False
    )
    parsed_args = parser.parse_args()

    if parsed_args.parameters is not None:
        for param in parsed_args.parameters.split(','):
            if len(param.split('=')) != 2:
                logging.error("'parameters' must be a comma separated list of <parameter>=<value>: %s", param)
                sys.exit(1)

    return parsed_args


def configure_logger(is_debug: bool) -> None:
    """
    Configures the logger.
    :param is_debug: True if the logger should be configured to print debug messages
    """
    if is_debug:
        # noinspection SpellCheckingInspection
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        # noinspection SpellCheckingInspection
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compute_configuration(params_to_overwrite: str) -> dict:
    """
    Computes the configuration overwriting the provided parameters.
    :param params_to_overwrite: A string containing the parameters to overwrite.
    :rtype: dict
    :return: The configuration as a dictionary
    """
    config_file = "default"

    # First look for configuration
    if params_to_overwrite is not None:
        for param in params_to_overwrite.split(','):
            if param.split("=")[0] == "configuration":
                config_file = param.split('=')[1]
                break

    # Load configuration
    parameters = load_configuration(config_file + ".xml")["tensorflow_configuration"]

    # Overwrite parameters
    if params_to_overwrite is not None:
        for param in params_to_overwrite.split(","):
            tokens = param.split("=")
            parameters[tokens[0]] = tokens[1]

    return parameters


def load_configuration(config_name: str) -> {}:
    """
    Load in a dictionary the configuration file at the provided path.
    :param config_name: The name of the configuration file.
    :rtype: {}
    :return: The dict representation of the configuration file.
    """
    # The absolute path of the current file
    script_path = os.path.realpath(__file__)

    # The root directory of the script
    script_dir = os.path.dirname(script_path)

    # The absolute path of the configuration directory
    cfg_dir = os.path.join(script_dir, "tf", "confs")
    logging.info("config directory is %s", cfg_dir)

    # Check the cfg_dir exists
    if not os.path.exists(cfg_dir):
        logging.error("config directory %s does not exist", cfg_dir)
        sys.exit(1)

    # Check if xml file of the conf exist
    xml_file = os.path.join(cfg_dir, config_name)
    if not os.path.exists(xml_file):
        logging.error("XML file %s not found", xml_file)
        sys.exit(1)

    # Load XML file
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read(), force_list={'input_class'})
    return doc


# noinspection PyShadowingNames
def validate_configuration(config_param: {}) -> None:
    """
    Validates the configuration file.
    :param config_param: The configuration xml converted to a dictionary.
    """
    # Checking input classes
    __input_classes = config_param.get("input_classes")
    if __input_classes is None:
        logging.error("inputs tag not found in %s.xml", args.configuration)
        sys.exit(1)

    if __input_classes.get("local_path") is None:
        logging.error("local_path is not specified in input_classes")
        sys.exit(1)

    # Collect information about remote position of inputs
    __remote_location = __input_classes.get("remote_location")
    if __remote_location is None:
        logging.error("remote_position is not specified in input classes")
        sys.exit(1)

    if __remote_location.get("user") is None:
        logging.error("user not set in remote_position of input classes")
        sys.exit(1)

    if __remote_location.get("key") is None:
        logging.error("key not set in remote position of input classes")
        sys.exit(1)

    if __remote_location.get("host") is None:
        logging.error("host not set in remote position of input classes")
        sys.exit(1)

    if __remote_location.get("path") is None:
        logging.error("path not set in remote position of input classes")
        sys.exit(1)

    # Get number of classes
    __num_classes = int(config_param.get("num_classes"))
    if __num_classes is None or __num_classes < 2:
        logging.error("Number of classes not set in xml file")
        sys.exit(1)

        # Get network type
    if config_param.get("network_type") is None:
        logging.error("Network type not set in xml file")
        sys.exit(1)

    # Get number of epochs
    if config_param.get("epochs_number") is None:
        logging.error("Epoch number not set in xml file")
        sys.exit(1)

    # Get batch size
    if config_param.get("batch_size") is None:
        logging.error("Batch size not set in xml file")
        sys.exit(1)

    # Get momentum
    if config_param.get("momentum") is None:
        logging.error("Momentum not set in xml file")
        sys.exit(1)

    # Get j
    if config_param.get("j") is None:
        logging.error("J not set in xml file")
        sys.exit(1)


def create_experiment_paths(local_path_param: str) -> (str, str):
    """
    Creates the directories needed for the experiment.
    :param local_path_param: The local_path_param element in the configuration.
    :rtype: (str, str)
    :return: A tuple containing the training path and the validation path
    """
    # If it does not exist, create local path root
    if not os.path.exists(local_path_param):
        os.makedirs(local_path_param)

    # If it does not exist, create train directory
    local_train_path = os.path.join(local_path_param, "train")
    if not os.path.exists(local_train_path):
        os.makedirs(local_train_path)

    # If it does not exist, create val directory
    local_val_path = os.path.join(local_path_param, "val")
    if not os.path.exists(local_val_path):
        os.makedirs(local_val_path)

    return local_train_path, local_val_path


def sync_classes(input_classes: {}, num_classes, train_path_param, val_path_param) -> None:
    """
    Synchronizes the remote paths with the local one.
    :param input_classes: The input classes.
    :param num_classes: The number of classes to train.
    :param train_path_param: The train path.
    :param val_path_param: The validation path.
    """
    added_classes = 0
    remote_location = input_classes["remote_location"]

    for input_class in input_classes["input_class"]:
        if added_classes == int(num_classes):
            break

        added_classes += 1
        logging.info("Found input class %s", input_class)
        # Check if the input class files are already available locally
        if os.path.exists(os.path.join(train_path_param, input_class)) and \
                os.path.exists(os.path.join(val_path_param, input_class)):
            continue

        rsync_command = "rsync -a -e \"ssh -i {} -o StrictHostKeyChecking=no\" {}@{}:{} {}".format(
            remote_location["key"],
            remote_location["user"],
            remote_location["host"],
            os.path.join(remote_location["path"], "train", input_class),
            train_path_param
        )
        logging.info("rsync command is %s", rsync_command)
        subprocess.call(rsync_command, shell=True, executable="/bin/bash")

        rsync_command = "rsync -a -e \"ssh -i {} -o StrictHostKeyChecking=no\" {}@{}:{} {}".format(
            remote_location["key"],
            remote_location["user"],
            remote_location["host"],
            os.path.join(remote_location["path"], "val", input_class),
            val_path_param
        )
        logging.info("rsync command is %s", rsync_command)
        subprocess.call(rsync_command, shell=True, executable="/bin/bash")


def create_temp_dirs(input_classes: {}, num_classes) -> (str, str, str):
    """
    Creates temporary directories and symlinks them with the training path and the validation path.
    :param input_classes: The input classes.
    :param num_classes: The number of classes to train.
    :rtype: (str, str, str)
    :return: A tuple containing the temporary directory, the temporary training path and the temporary validation path
    """
    temporary_directory = tempfile.mkdtemp()
    temporary_train = os.path.join(temporary_directory, "train")
    os.makedirs(temporary_train)

    temporary_val = os.path.join(temporary_directory, "val")
    os.makedirs(temporary_val)

    added_classes = 0
    for input_class in input_classes["input_class"]:
        os.symlink(os.path.join(train_path, input_class), os.path.join(temporary_train, input_class))
        os.symlink(os.path.join(val_path, input_class), os.path.join(temporary_val, input_class))

        added_classes += 1
        if added_classes == int(num_classes):
            break

    return temporary_directory, temporary_train, temporary_val


def calculate_files_number(input_classes: {}, num_classes, temporary_train, temporary_val) -> (int, int):
    """
    :param input_classes: The input classes.
    :param num_classes: The number of classes to train.
    :param temporary_train: The temporary training directory
    :param temporary_val: The temporary validation directory
    :rtype: (int, int)
    :return: A tuple containing the number of training file and the number of validation file.
    """
    training_files_number = 0
    validation_files = 0
    added_classes = 0

    for input_class in input_classes["input_class"]:
        training_files_number += len(os.listdir(os.path.join(temporary_train, input_class)))
        validation_files += len(os.listdir(os.path.join(temporary_val, input_class)))
        added_classes += 1
        if added_classes == int(num_classes):
            break

    return training_files_number, validation_files


def create_tf_records(root_dir, train_dir, val_dir, out_dir, n_classes) -> str:
    """
    Creates the required TFRecords from the images.
    :param root_dir: The root directory
    :param train_dir: The train directory
    :param val_dir: The validation directory
    :param out_dir: The output directory
    :param n_classes: The number of classes
    :return: The path of the records
    :rtype: str
    """
    build_script_dir = os.path.join(root_dir, "tf", "slim", "datasets")
    build_script_file = os.path.join(build_script_dir, "build_imagenet_data.py")

    records = os.path.expanduser(os.path.join(out_dir, str(n_classes)))

    if os.path.exists(records):
        return records

    os.makedirs(records)

    build_script_command = "python3 {} "\
                           "--train_directory={} " \
                           "--validation_directory={} " \
                           "--output_directory={} " \
                           "--imagenet_metadata_file={} " \
                           "--labels_file={} " \
                           "--bounding_box_file={}" \
            .format(
                build_script_file,
                train_dir,
                val_dir,
                records,
                "imagenet_metadata.txt",
                "imagenet_lsvrc_2015_synsets.txt",
                "imagenet_2012_bounding_boxes.csv"
            )

    subprocess.call(build_script_command, shell=True, executable="/bin/bash", cwd=build_script_dir)

    return records


def dump_conf(cfg) -> str:
    """
    Dumps the configuration dict to an xml file.
    :param cfg: The configuration dict
    :return: The generated xml
    :rtype: str
    """
    dicttoxml.set_debug(False)
    xml = parseString(
        dicttoxml.dicttoxml(
            cfg,
            custom_root="tensorflow_configuration",
            attr_type=False)
    ).toprettyxml(indent="    ")
    generated_xml_file = open("configuration.xml", "w")
    generated_xml_file.write(xml)
    generated_xml_file.close()

    return xml


def process_stdout(stdout: IO) -> (float, str, str, int, int, int):
    overall_execution_time = 0.0
    temp_starting_time = ""
    temp_ending_time = ""
    temp_starting_timestamp = 0
    temp_ending_timestamp = 0
    training_files_number = 0

    for line in stdout:
        if line.startswith("Overall training time is"):
            match = re.search(
                r'Overall training time is (?P<ttime>\d+.\d+) seconds \((?P<sdate>.+?) ==> (?P<edate>.+)\)',
                line
            )

            overall_execution_time = float(match.group('ttime'))
            starting_date_str = match.group('sdate')
            ending_date_str = match.group('edate')

            temp_starting_time = (datetime.datetime.strptime(starting_date_str, "%a %b %d %H:%M:%S %Y")) \
                .strftime("%Y-%m-%d %H-%M-%S")
            temp_ending_time = (datetime.datetime.strptime(ending_date_str, "%a %b %d %H:%M:%S %Y")) \
                .strftime("%Y-%m-%d %H-%M-%S")
            temp_starting_timestamp = int(datetime.datetime.strptime(starting_date_str, "%a %b %d %H:%M:%S %Y").timestamp())
            temp_ending_timestamp = int(datetime.datetime.strptime(ending_date_str, "%a %b %d %H:%M:%S %Y").timestamp())
        elif line.startswith("Number of training files is"):
            training_files_number = int(line.split()[5])

    stdout.close()

    return overall_execution_time, temp_starting_time, temp_ending_time, temp_starting_timestamp, temp_ending_timestamp, \
           training_files_number


def calculate_epoch(iteration_number: int, batch_size: int, train_size_param: int, total_epochs: int) -> int:
    epoch = int(batch_size * iteration_number / train_size_param) + 1

    if epoch > total_epochs:
        epoch -= 1

    return epoch


def collect_data(repetition_path, gpu_type, gpu_number, debug):
    try:
        # The iterations fractions
        iteration_fractions = [0.25, 0.50, 0.75]

        # The number of initial iterations to be skipped
        skipped_initial_iterations = 20

        # The root directory of the script
        global abs_root

        # Skip experiment for boot GPU
        if repetition_path.find("alexnet_cl_3_im_9999_ep_1_bs_256_mo_0.9") != -1:
            return
        csv_file_name = "tf.csv"
        if os.path.exists(csv_file_name):
            csv_file = open(csv_file_name, "a")
        else:
            csv_file = open(csv_file_name, "w")
            column_labels = [
                    "starting timestamp",
                    "starting time",
                    "tensorflow version",
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
                    "CPUs usage",
                    "Average CPU usage",
                    "GPUs usage",
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
        if not os.path.exists("tf_csvs"):
            os.mkdir("tf_csvs")

        execution_stdout_filename = os.path.join(repetition_path, "execution_stdout")
        execution_stderr_filename = os.path.join(repetition_path, "execution_stderr")
        hw_configuration_filename = os.path.join(repetition_path, "hw_configuration")
        if os.stat(execution_stdout_filename).st_size == 0:
            return
        if not os.path.exists(hw_configuration_filename):
            return
        execution_stdout = open(execution_stdout_filename)
        execution_stderr = open(execution_stderr_filename)
        hw_configuration = open(hw_configuration_filename)
        training_files_number = ""
        local_starting_timestamp = ""
        training_time = 0.0
        data_time = 0.0
        current_iteration = 0
        initial_training_time = 0.0
        initial_data_time = 0.0
        for line in execution_stdout:
            if line.startswith("Overall training time is"):
                overall_execution_time = line.split()[4]
                times_vector = re.split(r'\(|=|>|\)', line)
                local_starting_timestamp = int((datetime.datetime.strptime(times_vector[1].strip(), "%a %b %d %H:%M:%S %Y") - datetime.datetime(1970, 1, 1)).total_seconds())
                local_starting_time = (datetime.datetime.strptime(times_vector[1].strip(), "%a %b %d %H:%M:%S %Y")).strftime("%Y-%m-%d %H:%M:%S")
                local_ending_time = (datetime.datetime.strptime(times_vector[4].strip(), "%a %b %d %H:%M:%S %Y")).strftime("%Y-%m-%d %H:%M:%S")
            if line.startswith("Number of training files is"):
                training_files_number = line.split()[5]

        # Checking stderr for error
        error_found = False
        for line in execution_stderr:
            if line.find("THCudaCheck FAIL") != -1:
                error_found = True
                break
            elif line.find("RuntimeError: CUDA error: out of memory") != -1:
                error_found = True
                break
            elif line.find("Traceback (most recent call last)") != -1:
                error_found = True
                break
        if error_found:
            csv_file.close()
            return
        xml_configuration_file_name = os.path.join(repetition_path, "configuration.xml")
        if not os.path.exists(xml_configuration_file_name):
            csv_file.close()
            return
        if local_starting_timestamp == "":
            csv_file.close()
            return
        xml_configuration = xmltodict.parse(open(xml_configuration_file_name).read(), force_list={'input_class'})[
            "tensorflow_configuration"]
        configuration_gpus_number = xml_configuration.get("gpus_number")
        if configuration_gpus_number != None:
            gpu_number = int(configuration_gpus_number)

        num_classes = xml_configuration["num_classes"]
        user_batch_size = xml_configuration["batch_size"]
        batch_size = int(user_batch_size) * int(gpu_number)

        # Computing training_time
        for line in open(execution_stderr_filename):
            if line.startswith("INFO:tensorflow:global step "):
                match = re.search(r'INFO:tensorflow:global step (?P<step>\d+): loss = (\d+.\d+|nan) '
                                  r'\((?P<sec_step>\d+.\d+) sec/step\)', line)
                step = int(match.group('step'))
                sec_step = float(match.group('sec_step'))

                epoch = calculate_epoch(
                    step,
                    int(batch_size),
                    int(training_files_number),
                    int(xml_configuration["epochs_number"])
                )

                training_time += sec_step
                if current_iteration < skipped_initial_iterations:
                    initial_training_time = initial_training_time + float(sec_step)
                current_iteration = current_iteration + 1


        # Prepering csv file with details about iteration time executions
        repetition_number = os.path.basename(repetition_path)
        configuration_path = os.path.basename(os.path.dirname(os.path.dirname(repetition_path)))
        experiment_path = os.path.basename(os.path.dirname(repetition_path))
        end_information = False
        iterations_number = 0
        iteration_file_name = os.path.join("tf_csvs", "iterations_" + gpu_type.replace(" ", "-") + "_" + str(
            gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(local_starting_timestamp) + ".csv")
        if not os.path.exists(iteration_file_name):
            iteration_file = open(iteration_file_name, "w")
            iteration_file.write("Epoch,Phase,Iteration,DataTime,TrainingTime,Testtime,End\n")
            for line in open(execution_stderr_filename):
                if line.startswith("INFO:tensorflow:global step "):
                    match = re.search(r'INFO:tensorflow:global step (?P<step>\d+): loss = (\d+.\d+|nan) '
                                      r'\((?P<sec_step>\d+.\d+) sec/step\)', line)

                    step = int(match.group('step'))
                    sec_step = float(match.group('sec_step'))

                    epoch = calculate_epoch(
                        step,
                        int(batch_size),
                        int(training_files_number),
                        int(xml_configuration["epochs_number"])
                    )

                    iteration_data_time = "NaN"
                    end = "NaN"

                    data_time = data_time + float(iteration_data_time)
                    if current_iteration < skipped_initial_iterations:
                        initial_training_time = initial_training_time + float(sec_step)
                        initial_data_time = initial_data_time + float(iteration_data_time)
                    current_iteration = current_iteration + 1

                    iteration_file.write(str(epoch) + ",Training," + str(step) + "," + iteration_data_time + "," + str(
                        sec_step) + ",NaN," + end + "\n")
                    iterations_number = iterations_number + 1
            iteration_file.close()
        else:
            for line in open(iteration_file_name):
                if line.find("Training") != -1:
                    iterations_number = iterations_number + 1
        if os.path.exists(iteration_file_name) and not os.path.exists(iteration_file_name.replace(".csv", ".pdf")):
            create_graph_command = os.path.join(abs_root, "pytorch", "generate_iteration_graph.py") + " " + iteration_file_name + " -o " + iteration_file_name.replace(".csv", ".pdf") + " -s" + repetition_path
            logging.debug("Executing %s", create_graph_command)
            subprocess.call(create_graph_command, shell=True, executable="/bin/bash")

        # Preparing csv file with CPU and GPU utilization
        profile_CPU_output_filename = os.path.join(repetition_path, "profile_CPU_output")
        profile_GPU_output_filename = os.path.join(repetition_path, "profile_GPU_output")
        profile_file_name_cpu = os.path.join("tf_csvs", "profile_CPU_" + gpu_type.replace(" ", "-") + "_" + str(
            gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(local_starting_timestamp) + ".csv")
        profile_file_name_sum_cpu = os.path.join("tf_csvs", "profile_sum_CPU_" + gpu_type.replace(" ", "-") + "_" + str(
            gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(local_starting_timestamp) + ".csv")
        if os.path.exists(profile_CPU_output_filename) and (
                not os.path.exists(profile_file_name_cpu) or not os.path.exists(profile_file_name_sum_cpu)):
            # The collected data
            cpu_data = {}
            cpu_sum_data = {}
            maximum_CPU_number = 0
            current_timestamp = ""
            previous_timestamp = ""
            # Analyzing profile_CPU_output
            for line in open(profile_CPU_output_filename, "r"):
                # New entry
                if line.find("%CPU %MEM ARGS") != -1:
                    previous_timestamp = current_timestamp
                    # Old pattern
                    if line.startswith("%CPU %MEM ARGS"):
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
                    # New pattern
                    else:
                        split = line.replace("\\n%CPU %MEM ARGS", "").split()
                        current_timestamp_readable = split[4] + " " + split[5]
                        current_timestamp = split[1]
                    logging.debug("Found timestamp %s (%s)", current_timestamp, current_timestamp_readable)
                    # Workaround: some profile logs contains data of next repetition
                    if current_timestamp_readable > local_ending_time:
                        break
                    current_CPU_number = 0
                    cpu_data[current_timestamp] = {}
                    cpu_data[current_timestamp]["readable_timestamp"] = current_timestamp_readable
                elif line.find("apps/tf/slim/train_image_classifier.py") != -1:
                    split = line.split()
                    CPU_usage = split[0]
                    cpu_data[current_timestamp]["CPU" + str(current_CPU_number)] = CPU_usage
                    if not "CPU" + str(current_CPU_number) in cpu_sum_data:
                        cpu_sum_data["CPU" + str(current_CPU_number)] = 0.0
                    if previous_timestamp != "":
                        cpu_sum_data["CPU" + str(current_CPU_number)] = cpu_sum_data["CPU" + str(current_CPU_number)] + (float(CPU_usage) * (float(current_timestamp) - float(previous_timestamp)) / 1000000000)
                    current_CPU_number = current_CPU_number + 1
                    if current_CPU_number > maximum_CPU_number:
                        maximum_CPU_number = current_CPU_number
            # Writing results in csv file
            profile_file = open(profile_file_name_cpu, "w")
            profile_file.write("timestamp,readable timestamp")
            for CPU_number in range(0, maximum_CPU_number):
                profile_file.write(",CPU" + str(CPU_number))
            profile_file.write("\n")
            for timestamp in sorted(cpu_data):
                profile_file.write(timestamp)
                current_data = cpu_data[timestamp]
                profile_file.write("," + current_data["readable_timestamp"])
                for CPU_number in range(0, maximum_CPU_number):
                    if "CPU" + str(CPU_number) in current_data:
                        profile_file.write("," + current_data["CPU" + str(CPU_number)])
                    else:
                        profile_file.write(",0")
                profile_file.write("\n")
            profile_file.close()
            # Writing sum results in csv file
            profile_sum_file = open(profile_file_name_sum_cpu, "w")
            for CPU_number in range(0, maximum_CPU_number):
                if CPU_number != 0:
                    profile_sum_file.write(",")
                profile_sum_file.write("CPU" + str(CPU_number))
            profile_sum_file.write("\n")
            for CPU_number in range(0, maximum_CPU_number):
                if CPU_number != 0:
                    profile_sum_file.write(",")
                profile_sum_file.write(str(cpu_sum_data["CPU" + str(CPU_number)]))
            profile_sum_file.write("\n")
            profile_sum_file.close()
        profile_file_name_gpu = os.path.join("tf_csvs", "profile_GPU_" + gpu_type.replace(" ", "-") + "_" + str(
            gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(local_starting_timestamp) + ".csv")
        profile_file_name_sum_gpu = os.path.join("tf_csvs", "profile_sum_GPU_" + gpu_type.replace(" ", "-") + "_" + str(
            gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(local_starting_timestamp) + ".csv")

        if os.path.exists(profile_GPU_output_filename) and (
                not os.path.exists(profile_file_name_gpu) or not os.path.exists(profile_file_name_sum_gpu)):
            # The collected data
            gpu_data = {}
            gpu_sum_data = {}
            # The maximum number of GPU used
            maximum_GPU_number = 0
            previous_timestamp = ""
            current_timestamp = ""
            # Analyzing profile_GPU_output
            for line in open(profile_GPU_output_filename, "r"):
                logging.debug("Read %s", line)
                split = line.split()
                # New pattern
                if line.find("Timestamp") != -1:
                    previous_timestamp = current_timestamp
                    current_timestamp = split[1]
                    current_timestamp_readable = split[4] + " " + split[5]
                    gpu_data[current_timestamp] = {}
                    gpu_data[current_timestamp]["readable_timestamp"] = current_timestamp_readable
                    current_GPU_number = 0
                    logging.debug("Found timestamp %s", current_timestamp)
                    # Workaround: some profile logs contains data of next repetition
                    if current_timestamp_readable > local_ending_time:
                        break
                    continue
                # Old pattern
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
                    current_GPU_number = 0
                    logging.debug("Found timestamp (Old pattern) %s", current_timestamp)
                    # Workaround: some profile logs contains data of next repetition
                    if current_timestamp_readable > local_ending_time:
                        break
                    continue
                if line.find("utilization") != -1:
                    continue
                # The line actually stores GPU usage information
                GPU_usage = split[0]
                memory_usage = split[2]
                if not current_timestamp in gpu_data:
                    gpu_data[current_timestamp] = {}
                gpu_data[current_timestamp]["GPU" + str(current_GPU_number)] = GPU_usage
                if not "GPU" + str(current_GPU_number) in gpu_sum_data:
                    gpu_sum_data["GPU" + str(current_GPU_number)] = 0.0
                if previous_timestamp != "":
                    gpu_sum_data["GPU" + str(current_GPU_number)] = gpu_sum_data["GPU" + str(current_GPU_number)] + (float(GPU_usage) * (float(current_timestamp) - float(previous_timestamp)) / 1000000000)
                gpu_data[current_timestamp]["GPU" + str(current_GPU_number) + "memory"] = memory_usage
                current_GPU_number = current_GPU_number + 1
                logging.debug("Found GPU utilization. Number of GPU updated to %d", current_GPU_number)
                if current_GPU_number > maximum_GPU_number:
                    maximum_GPU_number = current_GPU_number
            # Writing results in csv file
            profile_file = open(profile_file_name_gpu, "w")
            profile_file.write("timestamp,readable timestamp")
            for GPU_number in range(0, maximum_GPU_number):
                profile_file.write(",GPU" + str(GPU_number))
            for GPU_number in range(0, maximum_GPU_number):
                profile_file.write(",GPU" + str(GPU_number) + "memory")
            profile_file.write("\n")
            for timestamp in sorted(gpu_data):
                profile_file.write(timestamp)
                current_data = gpu_data[timestamp]
                profile_file.write("," + current_data["readable_timestamp"])
                for GPU_number in range(0, maximum_GPU_number):
                    if "GPU" + str(GPU_number) in current_data:
                        profile_file.write("," + current_data["GPU" + str(GPU_number)])
                    else:
                        profile_file.write(",0")
                for GPU_number in range(0, maximum_GPU_number):
                    if "GPU" + str(GPU_number) + "memory" in current_data:
                        profile_file.write("," + current_data["GPU" + str(GPU_number) + "memory"])
                    else:
                        profile_file.write(",0")
                profile_file.write("\n")
            profile_file.close()
            # Writing sum results in csv file
            profile_sum_file = open(profile_file_name_sum_gpu, "w")
            for GPU_number in range(0, maximum_GPU_number):
                if GPU_number != 0:
                    profile_sum_file.write(",")
                profile_sum_file.write("GPU" + str(GPU_number))
            profile_sum_file.write("\n")
            for GPU_number in range(0, maximum_GPU_number):
                if GPU_number != 0:
                    profile_sum_file.write(",")
                profile_sum_file.write(str(gpu_sum_data["GPU" + str(GPU_number)]))
            profile_sum_file.write("\n")
            profile_sum_file.close()
        profile_file_name = os.path.join("tf_csvs", "profile_" + gpu_type.replace(" ", "-") + "_" + str(
            gpu_number) + "_" + configuration_path + "_" + experiment_path + "_" + str(local_starting_timestamp) + ".pdf")

        if os.path.exists(profile_file_name_cpu) and os.path.exists(profile_file_name_gpu) and not os.path.exists(
                profile_file_name):
            create_graph_command = os.path.join(abs_root, "pytorch",
                                                "generate_profile_graph.py") + " -c" + profile_file_name_cpu + " -g" + profile_file_name_gpu + " -o" + profile_file_name + " -s" + repetition_path + " -t" + overall_execution_time
            if end_information:
                create_graph_command = create_graph_command + " -i" + iteration_file_name
            if debug:
                create_graph_command = create_graph_command + " -d"
            logging.debug("Executing %s", create_graph_command)
            create_graph_return_value = subprocess.call(create_graph_command, shell=True, executable="/bin/bash")
            if create_graph_return_value != 0:
                logging.error("Error in analyzing result of %s", repetition_path)
                sys.exit(-1)

        network_type = xml_configuration["network_type"]
        epochs_number = xml_configuration["epochs_number"]
        computed_iterations_number = int(xml_configuration.get("iteration_number"))
        if not computed_iterations_number:
            computed_iterations_number = str(int(training_files_number) * int(epochs_number) / int(batch_size))
        tf_version = xml_configuration.get("tensorflow_version")
        if not tf_version:
            tf_version = "unknown"
        j = xml_configuration.get("j")
        if not j:
            j = "4"

        #Retrieving mac address
        for line in hw_configuration:
           if line.find("serial: ") != -1:
              mac_address = line.split()[1]

        if mac_address == "":
           logging.error("mac address not found")
           sys.exit(1)

        #Retrieving machine information
        #Add host_scripts to the directories for python packages search
        host_scripts_path = os.path.join(abs_root, "..", "host_scripts")
        sys.path.append(host_scripts_path)
        collect_data_package = __import__("collect_data")
        machine_information = collect_data_package.get_machine_information(mac_address)
        if xml_configuration.get("system_UUID"):
            system_uuid = xml_configuration.get("system_UUID")
        else:
            system_uuid = machine_information["system_uuid"]
        machine_name = machine_information["machine_name"]
        disk_speed = machine_information["disk_speed"]
        gflops = machine_information["gflops"]

        only_load = "0"
        overlapped = "0"

        profile = xml_configuration.get("profile")
        if not profile or not os.path.exists(profile_file_name_sum_cpu) or not os.path.exists(
                profile_file_name_sum_gpu):
            profile = "NaN"
            CPU_usage = "NaN"
            average_CPU_usage = "NaN"
            GPU_usage = "NaN"
            average_GPU_usage = "NaN"
            usage_ratio = "NaN"
        else:
            CPU_usage = 0.0
            GPU_usage = 0.0
            cpu_row_1 = linecache.getline(profile_file_name_sum_cpu, 2)
            split = cpu_row_1.split(",")
            for token in split:
                CPU_usage = CPU_usage + float(token.replace("\n", ""))
            average_CPU_usage = CPU_usage / float(overall_execution_time)
            gpu_row_1 = linecache.getline(profile_file_name_sum_gpu, 2)
            split = gpu_row_1.split(",")
            for token in split:
                GPU_usage = GPU_usage + float(token.replace("\n", ""))
            average_GPU_usage = GPU_usage / float(overall_execution_time)
            usage_ratio = CPU_usage / GPU_usage

        # Computing the values of fractions of this experiment
        iteration_number_fractions = {}
        # data_time_fractions = {}
        training_time_fractions = {}
        if iteration_fractions:
            current_iteration = 0
            # current_aggregated_data_time = 0.0
            current_aggregated_training_time = 0.0
            iteration_fractions_index = 0
            current_iterations_fraction_number = int(
                round(iterations_number * iteration_fractions[iteration_fractions_index]))
            iteration_number_fractions[
                iteration_fractions[iteration_fractions_index]] = current_iterations_fraction_number
            for line in open(iteration_file_name):
                if line.find("Training,") != -1:
                    current_iteration = current_iteration + 1
                    split = line.split(",")
                    # current_date_time = float(split[3])
                    current_training_time = float(split[4])
                    # current_aggregated_data_time = current_aggregated_data_time + current_date_time
                    current_aggregated_training_time = current_aggregated_training_time + current_training_time
                    if current_iteration == current_iterations_fraction_number:
                        # data_time_fractions[
                        #     iteration_fractions[iteration_fractions_index]] = current_aggregated_data_time
                        training_time_fractions[
                            iteration_fractions[iteration_fractions_index]] = current_aggregated_training_time
                        iteration_fractions_index = iteration_fractions_index + 1
                        if iteration_fractions_index < len(iteration_fractions):
                            current_iterations_fraction_number = int(
                                round(iterations_number * iteration_fractions[iteration_fractions_index]))
                            iteration_number_fractions[
                                iteration_fractions[iteration_fractions_index]] = current_iterations_fraction_number
                        else:
                            break
        for iteration_fraction in iteration_fractions:
            iteration_number_fraction = iteration_number_fractions[iteration_fraction]
            epochs_number_fraction = str(float(epochs_number) * iteration_fraction)
            # data_time_fraction = data_time_fractions[iteration_fraction]
            data_time_fraction = "NaN"
            training_time_fraction = training_time_fractions[iteration_fraction]
            csv_file.write(
                str(local_starting_timestamp) + "," + local_starting_time + "," + tf_version + "," + system_uuid + "," + mac_address + "," + machine_name + "," + gpu_type + "," + gflops + "," + disk_speed + "," + str(gpu_number) + "," + network_type + ",NaN," + num_classes + "," + str(batch_size) + "," + str(iteration_fraction) + "," + str(iteration_number_fraction) + "," + str(float(computed_iterations_number) * iteration_fraction) + ",0," + epochs_number_fraction + "," + j + "," + profile + ",NaN,NaN,NaN,NaN,NaN," + only_load + "," + overlapped + "," + str(data_time_fraction) + "," + str(training_time_fraction) + ",NaN,NaN,NaN," + repetition_number + "," + repetition_path + "\n")
            if iteration_number_fraction > skipped_initial_iterations:
                csv_file.write(str(local_starting_timestamp) + "," + local_starting_time + "," + tf_version + "," + system_uuid + "," + mac_address + "," + machine_name + "," + gpu_type + "," + gflops + "," + disk_speed + "," + str(gpu_number) + "," + network_type + ",NaN," + num_classes + "," + str(batch_size) + "," + str(iteration_fraction) + "," + str(iteration_number_fraction - skipped_initial_iterations) + "," + str(float(computed_iterations_number) * iteration_fraction - skipped_initial_iterations) + "," + str(skipped_initial_iterations) + "," + epochs_number_fraction + "," + j + "," + profile + ",NaN,NaN,NaN,NaN,NaN," + only_load + "," + overlapped + "," + "NaN" + "," + str(training_time_fraction - initial_training_time) + ",NaN,NaN,NaN," + repetition_number + "," + repetition_path + "\n")

        # Writing full experiment data
        csv_file.write(str(local_starting_timestamp) + "," + local_starting_time + "," + tf_version + "," + system_uuid + "," + mac_address + "," + machine_name + "," + gpu_type + "," + gflops + "," + disk_speed + "," + str(gpu_number) + "," + network_type + ",NaN," + num_classes + "," + str(batch_size) + ",1.0," + str(iterations_number) + "," + str(computed_iterations_number) + ",0," + epochs_number + "," + j + "," + profile + "," + str(CPU_usage) + "," + str(average_CPU_usage) + "," + str(GPU_usage) + "," + str(average_GPU_usage) + "," + str(usage_ratio) + "," + only_load + "," + overlapped + "," + "NaN" + "," + str(training_time) + "," + "NaN" + "," + overall_execution_time + "," + "NaN" + "," + repetition_number + "," + repetition_path + "\n")
        if iterations_number > skipped_initial_iterations:
            csv_file.write(str(local_starting_timestamp) + "," + local_starting_time + "," + tf_version + "," + system_uuid + "," + mac_address + "," + machine_name + "," + gpu_type + "," + gflops + "," + disk_speed + "," + str(gpu_number) + "," + network_type + ",NaN," + num_classes + "," + str(batch_size) + ",1.0," + str(iterations_number - skipped_initial_iterations) + "," + str(float(computed_iterations_number) - skipped_initial_iterations) + "," + str(skipped_initial_iterations) + "," + epochs_number + "," + j + "," + profile + "," + str(CPU_usage) + "," + str(average_CPU_usage) + "," + str(GPU_usage) + "," + str(average_GPU_usage) + "," + str(usage_ratio) + "," + only_load + "," + overlapped + "," + "NaN" + "," + str(training_time - initial_training_time) + "," + "NaN" + "," + overall_execution_time + "," + "NaN" +"," + repetition_number + "," + repetition_path + "\n")
        csv_file.close()

    except:
        logging.error("Error in analyzing result of %s", repetition_path)
        raise


if __name__ == '__main__':
    import tensorflow as tf

    # Parsing input arguments
    args = parse_arguments()

    # Initializing logger
    configure_logger(args.debug)

    # Root tree node
    config = compute_configuration(args.parameters)

    # Validating the configuration
    validate_configuration(config)

    # Creating the experiment directories (if not exist)
    local_path = config["input_classes"]["local_path"]
    #Expand ~ when present
    if local_path.find("~") != -1:
        local_path = os.path.expanduser(local_path)
    tfrecord_path_base = config["input_classes"]["tfrecords_path"]
    train_path, val_path = create_experiment_paths(local_path)

    # Syncing the classes
    sync_classes(config["input_classes"], config["num_classes"], train_path, val_path)

    # Creating temporary directory with experiment input
    temp_dir, temp_train, temp_val = create_temp_dirs(config["input_classes"], config["num_classes"])

    train_size, val_size = calculate_files_number(config["input_classes"],
                                                  config["num_classes"],
                                                  temp_train, temp_val)

    records_path = create_tf_records(abs_root, temp_train, temp_val, tfrecord_path_base, config["num_classes"])

    # Computing number of iterations
    config["iteration_number"] = int(int(config["epochs_number"]) * train_size / int(config["batch_size"])) + 1

    # Adding tensorflow version
    if 'VERSION' in tf.__dict__:
       version = tf.__dict__['VERSION']
    elif '__version__' in tf.__dict__:
       version = tf.__dict__['__version__']
    else:
       version = "unknown"
    config["tensorflow_version"] = version

    #Adding system UUID
    if not os.path.exists("/etc/system_uuid"):
        logging.warning("/etc/system_uuid does not exists")
    else:
        uuid_line = open("/etc/system_uuid", "r").readline()
        if len(uuid_line.split()) != 2:
            logging.error("Error in loading uuid: %s", str(uuid_line.split()))
            sys.exit(1)
        uuid = uuid_line.split()[1]

        config["system_UUID"] = uuid

    # Dump configuration in xml
    generated_xml = dump_conf(config)

    #If the number of gpus is specified, uses it, otherwise leave default (all gpus will be used)
    gpus_number = config.get("gpus_number")
    if gpus_number is None:
        export_gpus_command = ""
    else:
        export_gpus_command = "CUDA_VISIBLE_DEVICES=" + ",".join(str(gpu) for gpu in list(range(0, int(gpus_number)))) + " "


    # Perform the actual nn training
    imagenet_script = os.path.join(abs_root, "tf", "slim", "train_image_classifier.py")

    imagenet_command = "{} python3 {} " \
                       "--log_every_n_steps={} " \
                       "--model_name={} " \
                       "--max_number_of_steps={} " \
                       "--batch_size={} " \
                       "--momentum={} " \
                       "--num_readers={} " \
                       "--num_preprocessing_threads={} " \
                       "--dataset_dir={} " \
                       "--num_classes={} " \
                       "--save_summaries_secs={} " \
                       "--save_interval_secs={} " \
                       "--optimizer={} " \
        .format(
            export_gpus_command,
            imagenet_script,
            1,
            config["network_type"],
            config["iteration_number"],
            config["batch_size"],
            config["momentum"],
            config["j"],
            config["j"],
            records_path,
            config["num_classes"],
            0,
            0,
            "momentum"
        )

    logging.info("imagenet command is %s", imagenet_command)

    overall_starting_time = time.time()
    return_value = subprocess.call(imagenet_command, shell=True, executable="/bin/bash")
    overall_ending_time = time.time()

    if not args.no_clean:
        if os.path.exists("/tmp/tfmodel"):
            shutil.rmtree("/tmp/tfmodel")

    # Dump information about experiment
    print("TensorFlow nn training")
    print("Network type is " + config["network_type"])
    print("Number of classes is " + str(config["num_classes"]))
    print("Number of training files is " + str(train_size))
    print("Overall training time is {} seconds ({} ==> {})".format(
        str(overall_ending_time - overall_starting_time),
        time.ctime(overall_starting_time),
        time.ctime(overall_ending_time)
    ))
    print("Configuration file is:")
    print(generated_xml)

    sys.exit(return_value)
