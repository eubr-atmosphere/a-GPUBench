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

import logging
import os
import subprocess
import sys

#The loaded configuration
config = None

#The target file containing the list of experiments to be run
list_file_name = None

#The user to be created on the vm
remote_user = ""

#The absolute path of the current file
abs_script = os.path.realpath(__file__)

#The root directory of the script
abs_root = os.path.dirname(abs_script)


def copy_list_to_target(list_file_name_param):
    global config
    global list_file_name
    global remote_user

    list_file_name = list_file_name_param

    if not config.has_section("inhouse"):
        logging.error("inhouse section missing in configuration file")
        sys.exit(1)
    if not config.has_option("inhouse", "address"):
        logging.error("inouse section has not address field in configuration file")
        sys.exit(1)

    #The private ssh key
    private_key = os.path.join(os.path.abspath(os.path.join(abs_root, "..")), "keys", "id_rsa")
    os.chmod(private_key, 0o600)

    #modifica local
    #rsync_command = "rsync -a -e \"ssh -i " + private_key + " -o StrictHostKeyChecking=no\" " + list_file_name + " " + remote_user + "@" + config["inhouse"]["address"] + ":/tmp"

    rsync_command = "rsync -a -e \"ssh " + "-o StrictHostKeyChecking=no\" " + list_file_name + " " + remote_user + "@" + config["local"]["address"] + ":/tmp"
    logging.info("rsync command is %s", rsync_command)
    cmd = subprocess.Popen(rsync_command, shell=True)
    retcode = cmd.wait()
    if retcode == 0:
        logging.info("rsync completed")
    else:
        logging.error("Error in SSH")
        sys.exit(-1)


def initialize(config_param, args):
    global config
    global remote_user

    config = config_param

    if not config.has_section("local"):
        logging.error("inhouse section missing in configuration file")
        sys.exit(1)
    if not config.has_option("local", "username"):
        logging.error("inouse section has not username field in configuration file")
        sys.exit(1)
    remote_user = config["local"]["username"]

def parse_args(parser):
    return

def run_experiment(args):
    global abs_root
    global config
    global remote_user

    extra_options = ""
    if args.profile:
        extra_options = extra_options + " --profile " + args.profile

    if args.mail:
        extra_options = extra_options + " --mail "+ args.mail
    #modifica local
    remote_command = "screen -d -m /home/" + remote_user + "/a-GPUBench/vm_scripts/launch_local_experiment.py -a " + args.application + " --parameters-list /tmp/" + os.path.basename(list_file_name) + extra_options + " --repetitions " + str(args.repetitions)
    #remote_command = "/home/" + remote_user + "/a-GPUBench/vm_scripts/launch_local_experiment.py -a " + args.application + " --parameters-list /tmp/" + os.path.basename(list_file_name) + extra_options + " --repetitions " + str(args.repetitions)
    logging.info("remote command is %s", remote_command)

    #modifica local
    #ssh_command = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i " + os.path.join(abs_root, "..", "keys", "id_rsa") + " " + remote_user + "@" + config["inhouse"]["address"] + " " + remote_command
    ssh_command = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null " + " " + remote_user + "@" + config["local"]["address"] + " " + remote_command
    logging.info("ssh command is %s", ssh_command)
    cmd = subprocess.Popen(ssh_command, shell=True)
    retcode = cmd.wait()
    if retcode == 0:
        logging.info("SSH completed")
    else:
        logging.error("Error in SSH")
        sys.exit(1)
