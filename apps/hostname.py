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
import os
import socket

def compute_configuration_name(cl_parameters):
    """
    Compute the configuration name. Since hostname does not take any parameter, there can be only one configuration named "no_parameters"

    Paramters
    ---------
    cl_parameters: str
        Added for generality with respect to other applications.

    Return
    ------
    str
        return "no_parameters"
    """
    return "no_parameters"

def collect_data(repetition_path, gpu_type, gpu_number, debug):
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
    csv_file_name = "hostname.csv"
    if os.path.exists(csv_file_name):
        csv_file = open(csv_file_name, "a")
    else:
        csv_file = open(csv_file_name, "w")
        csv_file.write("starting_time, hostname\n")
    starting_time = os.path.dirname(os.path.join(repetition_path, os.pardir, os.pardir))
    execution_stdout = open(os.path.join(repetition_path, "execution_stdout"))
    csv_file.write(starting_time + ", " + execution_stdout.readline())
    csv_file.close()

if __name__ == "__main__":
    print(socket.gethostname())
