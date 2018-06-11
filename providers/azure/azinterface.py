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
import json
import logging
import os
import subprocess
import sys
import time

#Execute a command with az-cli
def az_execute_command(command):
    logging.debug("az %s", command)
    cmd = subprocess.Popen("az " + command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = cmd.stdout.read()
    stderr = cmd.stderr.read()
    retcode = cmd.wait()
    if retcode != 0:
        logging.error("az command returns %d", retcode)
        logging.error("stdout is\n %s", str(stdout))
        logging.error("stderr is \n %s", str(stderr))
        sys.exit(-1)
    if stdout.decode("utf-8") == "true\n":
        fixed_output = "{\"value\" :  true}"
    elif stdout.decode("utf-8") == "false\n":
        fixed_output = "{\"value\" :  false}"
    elif stdout.decode("utf-8") == "":
        fixed_output = "{}"
    else:
        fixed_output = stdout.decode()
    logging.debug(fixed_output)
    json_output = json.loads(fixed_output)
    json_string = json.dumps(json_output, indent=3, separators=(',', ': '), sort_keys=True)
    logging.debug("\n%s", json_string)
    if "error" in json_output and json_output["error"] != None:
        logging.error("az command return error")
        json_string = json.dumps(json_output, indent=3, separators=(',', ': '), sort_keys=True)
        logging.error("\n%s")
        sys.exit(-1)
    return json_output

#Create a resource group
def az_group_create(location, resource_group_name):
    return_value = az_group_exists(resource_group_name + "_" + location)
    if return_value:
        return
    command = "group create --name " + resource_group_name + "_" + location + " --location " + location
    az_execute_command(command)

#Check if a resource group exists
def az_group_exists(resource_group_name):
    json_output = az_execute_command("group exists -n " + resource_group_name)
    return json_output["value"]

#List the existing group
def az_group_list():
    return az_execute_command("group list")

#Return the name of a subscription given the id
def az_subscritpion_name(id_string):
    if id_string != None and id_string != "":
        json_output = az_execute_command("account list")
        for subscription in json_output:
            if subscription["id"] == id_string or subscription["name"] == id_string:
                return subscription["name"]
        logging.error("Subscription %s not found", id_string)
        sys.exit(1)
    else:
        json_output = az_execute_command("account show")
        return json_output["name"]

#Create a vm
def az_vm_create(subscription_name, resource_group_name, location, size, image, user, vm_name_prefix):
    #The absolute path of the current script
    abs_script = os.path.realpath(__file__)

    #The root direcotry of the script
    abs_root = os.path.dirname(abs_script)

    #The dns name of the VM
    dns_name = (vm_name_prefix + subscription_name + size).replace("_", "").replace("standard", "").lower()

    #Dry run with --validate to see which is the error (if any)
    command = "vm create -n " + size.replace("_", "") + " -g " + resource_group_name + " --size " + size + " --image " + image + " --ssh-key-value " + os.path.join(abs_root, "..", "..", "keys", "id_rsa.pub") + " --admin-username " + user + " --public-ip-address-dns-name " + dns_name + " --validate"
    az_execute_command(command)

    #Actually create the vm
    command = "vm create -n " + size.replace("_", "") + " -g " + resource_group_name + " --size " + size + " --image " + image + " --ssh-key-value " + os.path.join(abs_root, "..", "..", "keys", "id_rsa.pub") + " --admin-username " + user + " --public-ip-address-dns-name " + dns_name
    az_execute_command(command)

    #Remove old host key
    command = "ssh-keygen -R " + dns_name + "." + location + ".cloudapp.azure.com"
    logging.info("Removing old host key: %s", command)
    subprocess.call(command, shell=True, executable="/bin/bash")

#Check if a vm exists
def az_vm_exists(resource_group_name, size):
    #Check if the resource group exists
    if not az_group_exists:
        return False
    command = "vm list -g " + resource_group_name
    return_value = az_execute_command(command)
    if not return_value:
        return False
    for vm in return_value:
        logging.debug("VM size is %s", vm["hardwareProfile"]["vmSize"])
        if vm["hardwareProfile"]["vmSize"] == size:
            return True
    return False

#Initialize a vm
def az_vm_initialize(subscription_name, group_name, location, size, vm_name_prefix, user):
    #The absolute path of the current script
    abs_script = os.path.realpath(__file__)

    #The root direcotry of the script
    abs_root = os.path.dirname(abs_script)

    #Copy repository to vm
    az_vm_rsync_to(os.path.abspath(os.path.join(abs_root, "..", "..")) + "/", subscription_name, location, size, "/home/" + user + "/a-GPUBench", vm_name_prefix, user)

    #Initialize the VM - run the install script
    command = "cd /home/" + user + " && sudo -H a-GPUBench/vm_scripts/install.sh 2>&1 | tee /home/" + user + "/installation_log"

    az_vm_run_command_invoke(group_name + "_" + location, size.replace('_', ''), command)

#Execute command on vm
def az_vm_run_command_invoke(resource_group_name, name, command):
    command = "vm run-command invoke --command-id RunShellScript -g " + resource_group_name + " -n " + name + " --scripts \"" + command + "\""
    logging.info("Executing %s", command)
    az_execute_command(command)

#Scp files to VM
def az_vm_rsync_to(input_file, subscription_name, location, size, destination_folder, vm_name_prefix, user):
    #The absolute path of the current script
    abs_script = os.path.realpath(__file__)

    #The root directory of the script
    abs_root = os.path.dirname(abs_script)

    #The private ssh key
    private_key = os.path.join(os.path.abspath(os.path.join(abs_root, "..", "..")), "keys", "id_rsa")
    os.chmod(private_key, 0o600)

    retry = 0
    success = False
    while retry < 10 and not success:
        hostname = vm_name_prefix + size.replace("_", "").replace("standard", "").lower() + "." + location + ".cloudapp.azure.com"
        hostname = (vm_name_prefix + subscription_name + size).replace("_", "").replace("standard", "").lower() + "." + location + ".cloudapp.azure.com"
        logging.info("rsync to " + hostname + " trial " + str(retry))
        rsync_command = "rsync -a -e \"ssh -i " + private_key + " -o StrictHostKeyChecking=no\" " + input_file + " " + user + "@" + hostname + ":" + destination_folder
        logging.info("Executing %s", rsync_command)
        cmd = subprocess.Popen(rsync_command, shell=True)
        retcode = cmd.wait()
        if retcode == 0:
            logging.info("rsync completed")
            success = True
        else:
            logging.info("rsync failed")
            retry = retry + 1
            time.sleep(30)
    if not success:
        logging.error("Failure in rsync to vm")
        sys.exit(1)

#Execute command on vm through ssh
def az_vm_ssh_command_invoke(subscription_name, location, size, command, vm_name_prefix, user):
    #The absolute path of the current script
    abs_script = os.path.realpath(__file__)

    #The root directory of the script
    abs_root = os.path.dirname(abs_script)

    #The private ssh key
    private_key = os.path.join(os.path.abspath(os.path.join(abs_root, "..", "..")), "keys", "id_rsa")
    os.chmod(private_key, 0o600)

    retry = 0
    success = False
    while retry < 10 and not success:
        hostname = (vm_name_prefix + subscription_name + size).replace("_", "").replace("standard", "").lower() + "." + location + ".cloudapp.azure.com"
        logging.info("ssh connection to " + hostname + " trial " + str(retry))
        ssh_command = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i " + private_key + " " + user + "@" + hostname + " " + command
        logging.info("Executing %s", ssh_command)
        cmd = subprocess.Popen(ssh_command, shell=True)
        retcode = cmd.wait()
        if retcode == 0:
            logging.info("SSH completed")
            success = True
        else:
            logging.info("SSH failed")
            retry = retry + 1
            time.sleep(30)
    if not success:
        logging.error("Failure in ssh connection to vm")
        sys.exit(1)


#Starts a vm
def az_vm_start(resource_group_name, size):
    #Check if the VM exists
    if not az_vm_exists(resource_group_name, size):
        logging.error("VM %s on %s does not exist, so it cannot be started", size, resource_group_name)
        sys.exit(1)
    command = "vm start -g " + resource_group_name + " -n " + size.replace("_", "")
    az_execute_command(command)

#Return the status of the vm
def az_vm_status(resource_group_name, size):
    if not az_group_exists(resource_group_name):
        return "unexistent group"
    if not az_vm_exists(resource_group_name, size):
        return "unexistent vm in group"
    command = "vm list -g " + resource_group_name + " -d"
    return_value = az_execute_command(command)
    if not return_value:
        return "unexistent"
    for vm in return_value:
        if vm["hardwareProfile"]["vmSize"] == size:
            return vm["powerState"]
    return "unexistent"

#Stop and deallocate a running vm
def az_vm_deallocate(resource_group_name, size):
    command = "vm deallocate -g " + resource_group_name + " -n " + size.replace("_", "")
    az_execute_command(command)
