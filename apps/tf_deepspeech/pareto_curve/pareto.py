import xmltodict
import os
import operator
from subprocess import Popen, PIPE
import math
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import json
from util.gpu import get_available_gpus
import time
from cycler import cycler
import collections

CONFS_FOLDER = "confs"
# CONFIGURATION_FILENAME = 'remote.xml'
CONFIGURATION_FILENAME = 'default.xml'


def load_configuration(config_name: str) -> {}:
    # The absolute path of the current file
    #script_path = os.path.realpath(__file__)
    script_path = os.path.realpath('__file__')

    # The root directory of the script
    script_dir = os.path.dirname(script_path)

    cfg_dir = os.path.join(script_dir, CONFS_FOLDER)

    xml_file = os.path.join(cfg_dir, config_name)

    # Load XML file
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
    return doc


# this is a integer discretized non-negative x axis

class xAxis:
    def __init__(self, inverse=False, initial_power=1, base=2, last_power = 2):
        self.base = base
        self.power = int(initial_power)
        self.last_power = int(last_power)
        if inverse:
            self.direction = -1
        else:
            self.direction = 1

    def __iter__(self):
        return self

    def invert(self):
        self.direction = -self.direction
        return

    def __str__(self):
        return str(int(self.i))

    def __next__(self):
        if self.power <= self.last_power:
            self.i = math.pow(self.base, self.power)
            self.power = self.power + self.direction
            i = self.i
            return self
        else:
            raise StopIteration()


def main():
    num_gpus = get_available_gpus()
    print('Number of GPUs available: ' + str(num_gpus))
    conf = load_configuration(CONFIGURATION_FILENAME)
    main_conf = conf['pareto_configuration']
    x_flag = main_conf['x_axis']
    max_x_power = main_conf['max_x_power']
    min_x_power = main_conf['min_x_power']
    y_flag = main_conf['y_axis']
    coeff_delta = 0.01
    delta = 1
    partial_exec_command = main_conf['exec_command']
    success_set = {}
    failure_set = {}
    ended_dict = {}
    success_json_filename = main_conf['success_json']
    failure_json_filename = main_conf['failure_json']
    ended_json_filename = main_conf['ended_json']
    batch_axis = xAxis(base=2, initial_power=min_x_power, last_power=max_x_power)
    cwd = main_conf['cwd_command']
    post_exec_command = main_conf['post_exec_command']

    p = 0
    r = int(main_conf['max_y'])
    q = (p+r)/2

    if os.path.isfile(success_json_filename):
        with open(success_json_filename) as json_read:
            success_set = json.load(json_read)
        json_read.close()

    if os.path.isfile(failure_json_filename):
        with open(failure_json_filename) as json_read:
            failure_set = json.load(json_read)
        json_read.close()

    for current_batch in batch_axis:
        real_current_batch = int(str(current_batch)) * num_gpus
        if str(real_current_batch) in ended_dict:
            continue
        if str(real_current_batch) in success_set:
            p = int(float(success_set[str(real_current_batch)]))
        if str(real_current_batch) in failure_set:
            r = int(float(failure_set[str(real_current_batch)]))
        delta = 1 + math.ceil(p * coeff_delta)
        while (r-p) > delta:
            q = (p+r)/2
            exec_command = partial_exec_command + ' ' + y_flag + ' ' + str(int(q)) + ' ' + x_flag + ' ' + str(current_batch)
            print("Current iteration: Batch size: " + str(real_current_batch) + " Current number of hidden layers: " + str(q))
            print("Current command : " + exec_command)
            # time.sleep(120)
            print('start')
            # proc = Popen(exec_command, shell=True, stdout=PIPE, cwd=cwd)
            proc = Popen(exec_command, shell=True, cwd=cwd)
            # stdout = proc.stdout.readlines()
            res = proc.wait()
            return_code = proc.returncode
            subprocess.call(post_exec_command, shell=True, executable="/bin/bash")
            success = -1
            if int(return_code) == 0:
                success = True
                p = q+1
                if os.path.isfile(success_json_filename):
                    with open(success_json_filename) as json_read:
                        success_set = json.load(json_read)
                    json_read.close()
                if str(real_current_batch) in success_set:
                    # Overwrite the highest succeeded number of hidden layer
                    if int(float(success_set[str(real_current_batch)])) < int(float(q)):
                        success_set[str(real_current_batch)] = q
                else:
                    success_set[str(real_current_batch)] = q
                with open(success_json_filename, 'w') as json_file:
                    json_file.write(json.dumps(success_set))
                json_file.close()
            else:
                success = False
                r = q-1
                failure_set[str(real_current_batch)] = q

                if os.path.isfile(failure_json_filename):
                    with open(failure_json_filename) as json_read:
                        failure_set = json.load(json_read)
                    json_read.close()
                if str(real_current_batch) in failure_set:
                    # Overwrite the lowest failed number of hidden layer
                    if int(float(failure_set[str(real_current_batch)])) > int(float(q)):
                        failure_set[str(real_current_batch)] = q
                else:
                    failure_set[str(real_current_batch)] = q
                with open(failure_json_filename, 'w') as json_file:
                    json_file.write(json.dumps(failure_set))
                json_file.close()
            print("Last iteration success: " + str(success) + " Batch size: " + str(real_current_batch) + " N hidden layer: " + str(q))
            delta = 1 + math.ceil(p * coeff_delta)

        if os.path.isfile(ended_json_filename):
            with open(ended_json_filename) as json_read:
                ended_dict = json.load(json_read)
            json_read.close()
        ended_dict[str(real_current_batch)] = 1
        with open(ended_json_filename, 'w') as json_file:
            json_file.write(json.dumps(ended_dict))
        json_file.close()

        p = 0
        #r = q

    print(success_set)

    plt = create_graph(success_set)
    plt.savefig('success_set.png')
    plt.show()
    return 0

def plot_json(filename):
    if os.path.isfile(filename):
        with open(filename) as json_read:
            set = json.load(json_read)
        json_read.close()
        for k, v in set.items():
            set[k] = int(float(v))
        plt = create_graph(set)
        plt.savefig(filename + '.png')
        # plt.show()
        return

def plot_multiple_json(filenames):
    plt.figure(figsize=(15, 15))
    for filename in filenames:
        plot_json(filename)
    plt.show()
    return

def create_graph(dict):
    sorted_keys = sorted(dict, key=lambda value: int(float(value)))

    array = None

    for k in sorted_keys:
        k = int(k)
        v = dict[str(k)]
        v = int(v)
        new = np.array([[k, v]])
        if array is None:
            array = new
        else:
            array = np.concatenate((array, new), axis=0)


    x, y = array.T
    # plt.plot(x,y,'ro')
    plt.xlabel('train_batch_size')
    plt.ylabel('n_hidden')
    # plt.plot(x,y,'r', marker='o')
    plt.rc('axes',prop_cycle=(cycler('color', ['r', 'g', 'b', 'y'])))
    plt.plot(x,y, marker='o')
    return plt


if __name__ == "__main__":
    main()