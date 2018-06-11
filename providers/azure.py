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
import sys

#The absolute path of the current file
abs_script = os.path.realpath(__file__)

#The root directory of the script
abs_root = os.path.dirname(abs_script)

sys.path.append(os.path.join(abs_root, "azure"))
azinterface = __import__("azinterface")

#The loaded configuration
config = None

#The name of the group of the created vms
group_name = None

#The location of the vm
location = None

#The azure image used to create the vm
image = None

#The target file containing the list of experiments to be run
list_file_name = None

#The user to be created on the vm
remote_user = None

#The type of VM to be used
size = None

#The subscription name
subscription_name = None

def copy_list_to_target(list_file_name_param):
    global config
    global list_file_name
    global location
    global remote_user
    global size
    global subscription_name
    list_file_name = list_file_name_param
    azinterface.az_vm_rsync_to(list_file_name, subscription_name, location, size, "/tmp", config["azure"]["prefix"], remote_user)

def initialize(config_param, args):
    global config
    global group_name
    global location
    global image
    global remote_user
    global size
    global subscription_name

    config = config_param

    if args.group_name:
        group_name = args.group
    else:
        group_name = config["azure"]["default_group_name"]

    if args.location:
        location = args.location
    else:
        location = config["azure"]["default_location"]

    if args.image:
        image = args.image
    else:
        image = config["azure"]["default_image"]

    remote_user = config["azure"]["username"]

    #Check required parameters
    if not args.size:
        logging.error("--size is mandatory for azure provider")
        sys.exit(1)
    size = args.size
    if not args.subscription:
        logging.error("Subscription must be set for azure vm")
        sys.exit(1)

    #Check if key files exists
    if not os.path.exists(os.path.join(abs_root, "..", "keys", "id_rsa")) or not os.path.exists(os.path.join(abs_root, "..", "keys", "id_rsa.pub")):
        logging.error("Please add id_rsa and id_rsa.pub in keys")
        sys.exit(1)

    #Check if ssmtp configuration file exists
    if args.mail:
        if not os.path.exists(os.path.join(abs_root, "..", "vm_scripts", "revaliases")) or not os.path.exists(os.path.join(abs_root, "..", "vm_scripts", "ssmtp.conf")):
            logging.error("--mail option cannot be used without ssmtp configuration files")
            sys.exit(1)

    sys.path.append(os.path.join(abs_root, "azure"))
    import create_vm
    #Set the subscription
    azinterface.az_execute_command("account set --subscription " + args.subscription)
    subscription_name = azinterface.az_subscritpion_name(args.subscription)

    #Create vm
    create_vm.create_vm(subscription_name, group_name, args.debug, location, args.size, image, True, args.reuse, remote_user, config["azure"]["prefix"])

def parse_args(parser):
    parser.add_argument("--group-name", help="Azure: The name of the resource group")
    parser.add_argument("--location", help="Azure: The cluster location")
    parser.add_argument("--size", help="Azure: The size (aka the type) of the VM to be created")
    parser.add_argument("--image", help="Azure: The image to be used during creation of VM")
    parser.add_argument("--not-shutdown", help="Azure: Do not shutdown the vm", default=False, action="store_true")
    parser.add_argument("--subscription", help="Azure: The subscription to be used")
    parser.add_argument("--reuse", help="Azure: If true, an already running VM is reused", default=False, action="store_true")

def run_experiment(args):
    global config
    global location
    global remote_user
    global size
    global subscription_name


    extra_options = ""
    if args.profile:
        extra_options = extra_options + " --profile " + args.profile

    if not args.not_shutdown:
        extra_options = extra_options + " --shutdown"

    if args.mail:
        extra_options = extra_options + " --mail "+ args.mail

    remote_command = "screen -d -m /home/" + remote_user + "/a-GPUBench/vm_scripts/launch_local_experiment.py -a " + args.application + " --parameters-list /tmp/" + os.path.basename(list_file_name) + extra_options + " --repetitions " + str(args.repetitions) + " --subscription " + subscription_name
    logging.info("remote command is %s", remote_command)
    azinterface.az_vm_ssh_command_invoke(subscription_name, location, size, remote_command, config["azure"]["prefix"], remote_user)
