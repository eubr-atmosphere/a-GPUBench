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
import logging
import os
import shutil
import socket
import subprocess
import sys
import time

"""
Script used to deallocate a Microsoft Azure VM

The script deallocates a specific Microsoft Azure VM
Arguments of the scripts are:
    --mail: the mail address to which the notification must be sent
    --name: the name of the VM to be deallocated
    -s, --subscription: the subscription to which the VM belongs
"""

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

parser = argparse.ArgumentParser(description="Deallocate all stopped VMs and wait for deallocating a specific VM")
parser.add_argument("--mail", help="The mail address to which the notification must be sent")
parser.add_argument("--name", help="The name of the VM to be waited", required=True)
parser.add_argument("-s", "--subscription", help="The Azure subscription to be used")

args = parser.parse_args()

#The absolute path of the current script
abs_script = os.path.abspath(sys.argv[0])

#The root directory of the script
abs_root = os.path.dirname(abs_script)

sys.path.append(os.path.join(abs_root, "..", "providers", "azure"))
azinterface = __import__("azinterface")

#Switch to subscripiton
if args.subscription != None:
    azinterface.az_execute_command("account set --subscription " + args.subscription)

return_command = azinterface.az_group_list()
for group in return_command:
    if "GPUTest" in group["name"]:
        logging.info("Examining vms in %s", group["name"])
        az_list = azinterface.az_execute_command("vm list -g " + group["name"])
        for vm in az_list:
            status = azinterface.az_vm_status(group["name"], vm["hardwareProfile"]["vmSize"])
            logging.info("   Status of " + vm["name"] + " is " + status)
            if status == "VM stopped":
                azinterface.az_vm_deallocate(group["name"], vm["hardwareProfile"]["vmSize"])
for group in return_command:
    if "GPUTest" in group["name"]:
        logging.info("Examining vms in %s", group["name"])
        az_list = azinterface.az_execute_command("vm list -g " + group["name"])
        for vm in az_list:
            if vm["name"] == args.name:
                status = azinterface.az_vm_status(group["name"], vm["hardwareProfile"]["vmSize"])
                counter = 0
                while status != "VM stopped" and status != "VM deallocated":
                    logging.info("Status of %s is %s - waiting other 60 seconds", args.name, status)
                    counter = counter + 1
                    if counter == 10:
                        if args.mail:
                            if shutil.which("ssmtp"):
                                mail_content = ""
                                mail_content = mail_content + "To: " + args.mail + "\n"
                                mail_content = mail_content + "From: " + os.getlogin() + "." + socket.getfqdn() + "\n"
                                mail_content = mail_content + "Subject: ALERT: Status of az vm " + args.name + " is " + status
                                subprocess.call("echo \"" + mail_content + "\" | ssmtp " + args.mail, shell=True, executable="/bin/bash")
                            elif shutil.which("mutt"):
                                subprocess.call("echo \"\" | mutt -s \"ALERT: Status of az vm " + args.name + " is " + status + "\" " + args.mail, shell=True, executable="/bin/bash")
                        logging.error("Deallocation of %s failed", args.name)
                        sys.exit(1)
                    time.sleep(60)
                    status = azinterface.az_vm_status(group["name"], vm["hardwareProfile"]["vmSize"])
                if status == "VM stopped":
                    azinterface.az_vm_deallocate(group["name"], vm["hardwareProfile"]["vmSize"])
                if args.mail:
                    if shutil.which("ssmtp"):
                        mail_content = ""
                        mail_content = mail_content + "To: " + args.mail + "\n"
                        mail_content = mail_content + "From: " +  os.getlogin() + "." + socket.getfqdn()+ "\n"
                        mail_content = mail_content + "Subject: az vm " + args.name + " correctly deallocated"
                        ssmtp_command = "echo \"" + mail_content + "\" | ssmtp " + args.mail
                        logging.info("ssmtp_command is %s", ssmtp_command)
                        subprocess.call(ssmtp_command, shell=True, executable="/bin/bash")
                    elif shutil.which("mutt"):
                        mutt_command = "echo \"\" | mutt -s \"az vm " + args.name + " correctly deallocated\" " + args.mail
                        logging.info("mutt command is %s", mutt_command)
                        subprocess.call(mutt_command, shell=True, executable="/bin/bash")
                    else:
                        logging.error("Mail cannot be sent")
                        sys.exit(1)
                sys.exit(0)
