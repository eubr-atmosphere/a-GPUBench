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
import configparser
import logging
import os
import sys
import time

import azinterface

def create_vm(subscription_name, group_name, debug, location, size, image, running, reuse, user, vm_name_prefix):
    #Create resource group (if it does not exist)
    azinterface.az_group_create(location, group_name)

    #Check if VM already exists
    already_exists = azinterface.az_vm_exists(group_name+ "_" + location, size)
    if already_exists:
        status = azinterface.az_vm_status(group_name+ "_" + location, size)
        if reuse:
            if status != "VM running":
                logging.error("Cannot reuse machine since its status is %s", status)
                sys.exit(-1)
        else:
            if status != "VM deallocated":
                logging.error("Machine exists and it is not deallocated but %s", status)
                sys.exit(1)
            else:
                logging.info("Machine exists and it is stopped and deallocated")
                azinterface.az_vm_start(group_name + "_" + location, size)
            command = "rm -rf /home/" + user + "/a-GPUBench"
            azinterface.az_vm_ssh_command_invoke(subscription_name, location, size, command, vm_name_prefix, user)
    else:
        azinterface.az_vm_create(subscription_name, group_name + "_" + location, location, size, image, user, vm_name_prefix)
    logging.info("Waiting 60 seconds for VM boot")
    status = azinterface.az_vm_status(group_name+ "_" + location, size)
    while status != "VM running":
        if status != "VM starting" and status != "unexistent":
            logging.error("Machine status is %s", status)
            sys.exit(1)
        logging.info("Status of VM is %s", status)
        time.sleep(10)
        status = azinterface.az_vm_status(group_name+ "_" + location, size)

    #Initialize the VM - run the install script
    azinterface.az_vm_initialize(subscription_name, group_name, location, size, vm_name_prefix, user)

    #If required stop and deallocate vm
    if not running:
        azinterface.az_vm_deallocate(group_name + "_" + location, size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create az VM")
    parser.add_argument('-g', "--group-name", help="The name of the resource group", default="GPUTest")
    parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    parser.add_argument('-s', "--size", help="The size (aka the type) of the VM to be created", required=True)
    parser.add_argument('-r', "--running", help="Do not deallocate the vm after the creation", default=False)
    parser.add_argument('-u', "--user", help="The user to be created in the vm", required=True)
    parser.add_argument("--subscription", help="The used subscription", required=True)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #The absolute path of the current script
    abs_script = os.path.abspath(sys.argv[0])

    #The root directory of the script
    abs_root = os.path.dirname(abs_script)

    #Read configuration file
    current_configuration_file_name = os.path.join(abs_root, "..", "configurations", "current.ini")
    config = configparser.ConfigParser()
    config.read(current_configuration_file_name)

    create_vm(args.subscription, args.group_name, args.debug, args.location, args.size, args.image, args.running, False, args.user, config["azure"]["prefix"])
