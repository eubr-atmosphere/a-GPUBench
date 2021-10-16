#!/usr/bin/python3
"""
Copyright 2018 Marco Lattuada
Copyright 2021 Giovanni Dispoto

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
import shutil
import subprocess
import sys
import tempfile
import xmltodict

import provider

class Provider(provider.Provider):
    """
    Class to manage the execution of experiments on the host

    Attributes
    ----------
    _abs_root: str
        The absolute path containing the root of the library"

    _list_file_name: str
        The file containing the list of experiments to be executed

    Methods
    -------
    run_experiment()
        Run the experiments

    copy_list_to_target()
        Copy the file with the list of experiments to be executed
    """

    #_abs_root = None

    #_list_file_name = None
    def __init__(self, config, args):
     #The absolute path of the current file
     super().__init__(config, args)
     abs_script = os.path.realpath(__file__)

     #The root directory of the script
     self._abs_root = os.path.dirname(abs_script)
     self._local_user = config["local"]["username"]


    def copy_list_to_target(self, list_file_name):
        """
        Copy the list_file_name in a temporary location

        Parameters
        ----------
        list_file_name: str
            The name of the file to be copied
        """
        #shutil.copyfile(list_file_name, os.path.join(tempfile.gettempdir(), os.path.basename(list_file_name)))
        self._list_file_name = list_file_name
        private_key = os.path.join(os.path.abspath(os.path.join(self._abs_root, "..")), "keys", "id_rsa")
        os.chmod(private_key, 0o600)
        
        rsync_command = "rsync -a -e \"ssh -i " + private_key + " -o StrictHostKeyChecking=no\" " + list_file_name + " " + self._local_user + "@" + self._config["local"]["address"] + ":~/Desktop/output/config"
        #I should open the xml and read what file to send to the target machine
        #After this, the target machine should use this file as configuration file
        cmd = subprocess.Popen(rsync_command, shell=True)

        f = open(list_file_name, "r")
        list_file_name_parsed = f.readline().split('=')[1]
        
        f.close()

        rsync_command = "rsync -a -e \"ssh -i " + private_key + " -o StrictHostKeyChecking=no\" " + "/container-data/src/apps/tf/confs/"+list_file_name_parsed+".xml" + " " + self._local_user + "@" + self._config["local"]["address"] + ":~/Desktop/output/config"
        logging.info("rsync command is %s", rsync_command)
        cmd = subprocess.Popen(rsync_command, shell=True)
        retcode = cmd.wait()
        if retcode == 0:
            logging.info("rsync completed")
        else:
            logging.error("Error in SSH rsync")
            sys.exit(-1)

    def run_experiment(self):
        """
        Run the experiments
        """
        extra_options = ""
        if self._args.profile:
            extra_options = extra_options + " --profile " + self._args.profile

        if self._args.mail:
            extra_options = extra_options + " --mail "+ self._args.mail

        if self._args.debug:
            extra_options = extra_options + " --debug"

        utility = __import__("utility")
        root_project = utility.get_project_root()

        command = "docker run -v /home/gio/Desktop/output:/data "+self._config["local"]["container-name"]+ " /bin/bash -c 'python3 ./data/src/vm_scripts/launch_local_experiment.py -a "+self._args.application +" --parameters-list /data/config/local_ex"+ extra_options+  " --output " + self._args.output+"'"
        logging.info("command is %s", command)
        ssh_command = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i " + os.path.join(os.path.abspath(os.path.join(self._abs_root, "..")), "keys", "id_rsa") + " " + self._local_user + "@" + self._config["local"]["address"] + " " + '\"'+ command + '\"'
        cmd = subprocess.Popen(ssh_command, shell=True)
        retcode = cmd.wait()
        if retcode == 0:
            logging.info("launched local experiment")
        else:
            logging.error("Error in launching local experiment")
            sys.exit(1)

def parse_args(parser):
    """
    Add to the command line parser the options related to the host (none so far)
    """
    return
