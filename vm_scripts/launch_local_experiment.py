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
import datetime
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

parser = argparse.ArgumentParser(description="Launch job on cluster")
parser.add_argument('-a', "--application", help="The application to be run. It must correspond to an execuatble or a script present in directory apps", required=True)
parser.add_argument('-o', "--output", help="The output root directory", default="output")
parser.add_argument('-p', "--parameters-list", help="The file containing the list of the parameters of the experiment", required=True)
parser.add_argument('-r', "--repetitions", help="The number of times the application has to be executed", default=1)
parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
parser.add_argument('-s', "--shutdown", help="Shutdown the vm", default=False, action="store_true")
parser.add_argument("--subscription", help="The subscription to be used (1 or 2)")
parser.add_argument("--mail", help="The mail address to which end notification must be sent")
parser.add_argument("--profile", help="Log resource usage", default="")

args = parser.parse_args()

#The absolute path of the current script
abs_script = os.path.abspath(sys.argv[0])

#The root directory of the script
abs_root = os.path.dirname(abs_script)

#Read configuration file
current_configuration_file_name = os.path.join(abs_root, "..", "configurations", "current.ini")
config = configparser.ConfigParser()
config.read(current_configuration_file_name)

#Data about logserver
logserver_address = config["logserver"]["address"]
logserver_username = config["logserver"]["username"]
logserver_group = config["logserver"]["group"]

if args.debug:
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if args.shutdown:
    if not args.subscription:
        logging.error("--subscription must be specified when --shutdown is enabled")
        sys.exit(1)


#The key
key_file = os.path.abspath(os.path.join(os.path.abspath(abs_root), "..", "keys", "id_rsa"))

#The absolute path of the applications directory
apps_dir = os.path.join(abs_root, "..", "apps")
logging.info("apps directory is %s", apps_dir)

#Check the apps_dir exists
if not os.path.exists(apps_dir):
    logging.error("Apps directory %s does not exist", apps_dir)
    sys.exit(1)


#Look for app
if os.path.exists(os.path.join(apps_dir, args.application + ".py")):
    application_file = os.path.join(apps_dir, args.application + ".py")
else:
    logging.error("App %s  not found in %s", args.application, apps_dir)
    sys.exit(1)

#Add app to include dir of python
sys.path.append(apps_dir)

#Import application
app_package = __import__(args.application)

#Create output directory
output_root = os.path.abspath(args.output)
if os.path.exists(output_root):
    shutil.rmtree(output_root)

#Adding hostname
output_hostname = os.path.join(output_root, socket.gethostname())

#Adding app
output_app = os.path.join(output_hostname, args.application)

#The temporary file storing the hw configuration
hw_configuration_file_name = tempfile.NamedTemporaryFile(delete=False).name

#Collecting information about the system
collect_command = "(uname -a && lsb_release -a && lshw) > " + hw_configuration_file_name + " 2>&1"
subprocess.call(collect_command, shell=True, executable="/bin/bash")

#Iterate on the lines of the parameters_list
parameters_list_file = open(args.parameters_list)

#The retcode
retcode = 0

for line in parameters_list_file.read().splitlines():

    #Let's application compute the configuration name
    configuration_name = app_package.compute_configuration_name(line)
    if args.profile != "":
        configuration_name = configuration_name + "_profile_" + args.profile

    #Adding configuration
    output_conf = os.path.join(output_app, configuration_name)

    #Adding timestamp
    current_time_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if os.path.exists(os.path.join(output_conf, current_time_string)):
        counter = 1
        while os.path.exists(os.path.join(output_conf, current_time_string + "_" + counter)):
            counter = counter + 1
        output_experiment = os.path.join(output_conf, current_time_string + "_" + counter)
    else:
        output_experiment = os.path.join(output_conf, current_time_string)

    os.makedirs(output_experiment)

    #Preparing experiment command
    if args.profile != "":
        line = line + ",profile=" + args.profile
    local_command = application_file + " --parameters " + line

    #Execute the experiment
    starting_time = time.ctime()
    for repetition in range(0, int(args.repetitions)):
        #Create directory for repetition
        output_repetition = os.path.join(output_experiment, str(repetition))
        os.makedirs(output_repetition)
        stdout_file_name = os.path.join(output_repetition, "execution_stdout")
        stderr_file_name = os.path.join(output_repetition, "execution_stderr")
        logging.info("Executing experiment: %s", local_command)
        wrapped_command = "{ { " + local_command + "; } > >(tee " + stdout_file_name + " ); } 2> >(tee " + stderr_file_name + " >&2)"
        if args.profile != "":
            profile_CPU_output_file_name = os.path.join(output_repetition, "profile_CPU_output")
            profile_GPU_output_file_name = os.path.join(output_repetition, "profile_GPU_output")
            profile_CPU_command = os.path.join(abs_root, "profile_CPU.sh") + " " + args.profile + " > " + profile_CPU_output_file_name + " 2>&1"
            profile_GPU_command = os.path.join(abs_root, "profile_GPU.sh") + " " + args.profile + " > " + profile_GPU_output_file_name + " 2>&1"
#         profile_output_file = open(profile_output_file_name, "w")
            profile_CPU_process = subprocess.Popen(profile_CPU_command, cwd=output_repetition, shell=True, executable="/bin/bash", preexec_fn=os.setsid)
            profile_GPU_process = subprocess.Popen(profile_GPU_command, cwd=output_repetition, shell=True, executable="/bin/bash", preexec_fn=os.setsid)
        local_retcode = subprocess.call(wrapped_command, cwd=output_repetition, shell=True, executable="/bin/bash")
        if args.profile != "":
            os.killpg(os.getpgid(profile_CPU_process.pid), signal.SIGKILL)
            os.killpg(os.getpgid(profile_GPU_process.pid), signal.SIGKILL)
        retcode = retcode | local_retcode #Use of local_retcode prevents short circuit
        shutil.copy2(hw_configuration_file_name, os.path.join(output_repetition, "hw_configuration"))
    ending_time = time.ctime()

    #Copy files
    rsync_command = "rsync -a --out-format=\"%n\" --ignore-existing -e \"ssh -i " + key_file + " -o StrictHostKeyChecking=no\" --chown " + logserver_username + ":" + logserver_group + " " + output_root + " " + logserver_username + "@" + logserver_address + ":/home/" + logserver_username
    logging.info("rsync command is %s", rsync_command)
    cmd = subprocess.Popen(rsync_command, shell=True, stdout=subprocess.PIPE, executable="/bin/bash")
    stdout = cmd.stdout.read()

    if retcode == 0:
        exit_status = "SUCCESS"
    else:
        exit_status = "FAILURE"

    if args.mail:
        mail_content = ""
        mail_content = mail_content + "To: " + args.mail + "\n"
        mail_content = mail_content + "From: " + os.getlogin() + "." + socket.getfqdn() + "\n"
        mail_content = mail_content + "Subject: Azure run on " + socket.gethostname() + " - " + args.application + " - " + configuration_name + " ended: " + exit_status + "\n"
        mail_content = mail_content + "\n"
        mail_content = mail_content + "Experiment started at " + starting_time + "\n"
        mail_content = mail_content + "Experiment ended at " + ending_time + "\n"
        mail_content = mail_content + "Results are available on " + logserver_address + ":\n"
        for stdout_line in stdout.decode("utf-8").split("\n"):
            if "execution_std" in stdout_line:
                mail_content = mail_content + "/home/" + logserver_username + "/" + line
        subprocess.call("echo \"" + mail_content + "\" | ssmtp " + args.mail, shell=True, executable="/bin/bash")

this_hostname = socket.gethostname()

#Prepare deallocation
if args.shutdown:
    if args.mail:
        mail_arg = " --mail " + args.mail
    else:
        mail_arg = ""
    deallocate_command = "ssh -i " + key_file + " -o StrictHostKeyChecking=no " + logserver_username + "@" + logserver_address + " 'nohup /home/" + logserver_username + "/a-GPUBench/host_scripts/deallocate.py --name=" + this_hostname + mail_arg + " --subscription " + args.subscription + " >/home/" + logserver_username + "/deallocation_log 2>&1 &'"
    logging.info("deallocation command is %s", deallocate_command)
    subprocess.call(deallocate_command, shell=True, executable="/bin/bash")

#Shutdown
if args.shutdown:
    shutdown_command = "sudo shutdown -h now"
    logging.info("Shutdown command is %s", shutdown_command)
    subprocess.call(shutdown_command, shell=True, executable="/bin/bash")
