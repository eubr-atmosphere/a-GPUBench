"""
Copyright 2018-2019 Marco Lattuada
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
import subprocess
import sys
import time

import provider

class Provider(provider.Provider):
    """
    Class to manage the execution of experiments on a remote server accessed through ssh

    Attributes
    ----------
    _abs_root: str
        The absolute path containing the root of the library"

    Methods
    -------
    run_experiment()
        Run the experiments on the remote server

    copy_list_to_target()
        Copy the file with the list of experiments to be executed on the remote server
    """
    _abs_root = None

    def __init__(self, config, args):
        """
        Arguments
        ---------
        config: dict of str: dict of str: str
            The dictionary created from configuration file

        args
            The command line arguments parsed
        """
        super().__init__(config, args)
        if not config.has_section("inhouse"):
            logging.error("inhouse section missing in configuration file")
            sys.exit(1)
        if not config.has_option("inhouse", "username"):
            logging.error("inouse section has not username field in configuration file")
            sys.exit(1)
        self._remote_user = config["inhouse"]["username"]

        #The absolute path of the current file
        abs_script = os.path.realpath(__file__)

        #The root directory of the script
        self._abs_root = os.path.dirname(abs_script)

    def copy_list_to_target(self, list_file_name):
        """
        Copy the list_file_name to target

        Parameters
        ----------
        list_file_name: str
            The name of the file to be copied
        """
        self._list_file_name = list_file_name

        if not self._config.has_section("inhouse"):
            logging.error("inhouse section missing in configuration file")
            sys.exit(1)
        if not self._config.has_option("inhouse", "address"):
            logging.error("inouse section has not address field in configuration file")
            sys.exit(1)

        #The private ssh key
        private_key = os.path.join(os.path.abspath(os.path.join(self._abs_root, "..")), "keys/", self._config['inhouse']['machine-name'], "id_rsa")
        os.chmod(private_key, 0o600)

        rsync_command = " rsync -a -e \" sshpass -f /container-data/data/ssh_pwd ssh -i " + private_key + " -o StrictHostKeyChecking=no\" " + list_file_name + " " + self._remote_user + "@" + self._config["inhouse"]["address"] + ":storage/data/config"
        cmd = subprocess.Popen(rsync_command, shell=True)
        retcode = cmd.wait()

        #I should open the xml and read what file to send to the target machine
        #After this, the target machine should use this file as configuration file
        

        f = open(list_file_name, "r")
        list_file_name_parsed = f.readline().split('=')[1]
        
        f.close()

        rsync_command = "rsync -a -e \" sshpass -f /container-data/data/ssh_pwd  ssh -i " + private_key + " -o StrictHostKeyChecking=no\" " + "/container-data/src/apps/tf/confs/"+list_file_name_parsed+".xml" + " " + self._remote_user + "@" + self._config["inhouse"]["address"] + ":storage/data/config"
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
        
        #
        #docker run -it test /bin/bash -c "cd /root/app && pip3 install -r requirements.txt && python3.7 ./vm_scripts/launch_local_experiment.py -a tf --parameters-list local_ex  -o ./tf_out --mail dispoto97@gmail.com --repetitions 1" 
        #
        #remote_command = "screen -d -m /home/" + self._remote_user + "/a-GPUBench/vm_scripts/launch_local_experiment.py -a " + self._args.application + " --parameters-list /tmp/" + os.path.basename(self._list_file_name) + extra_options + " --repetitions " + str(self._args.repetitions) + " --output " + self._args.output
        #remote_command = "cd Desktop/ResearchProject/ && docker run -v /home/gio/Desktop/output:/data test /bin/bash -c 'cd /root/app  && python3 ./vm_scripts/launch_local_experiment.py -a "+self._args.application +" --parameters-list apps/tf/confs/local_ex"+ extra_options+  " --repetitions " + str(self._args.repetitions) + " --output " + self._args.output+" && cp -r "+self._args.output+"/* /data/"+self._args.output.split("/")[1]+"'"
        remote_command = "cd storage/building_enviroment &&  NV_GPU=5 nvidia-docker run --user 1060:1060 --rm --name "+ self._config["inhouse"]["container-name"]+"_" + str(time.time()) +" -v ~/storage/data/:/data "+self._config["inhouse"]["container-name"]+" /bin/bash -c 'python3 ./data/src/vm_scripts/launch_local_experiment.py -a "+self._args.application +" --parameters-list /data/config/"+self._list_file_name.split("/")[-1] + extra_options+  " --output " + self._args.output+"'"# + "&& cp -r "+self._args.output+"/* /data/"+self._args.output.split("/")[1]+"'"
        #remote_command = "cd storage/building_enviroment &&  NV_GPU=0 nvidia-docker run --user 1060:1060 -v ~/storage/data/:/data dispoto /bin/bash -c 'ls'"
        logging.info("remote command is %s", remote_command)

        ssh_command = "sshpass -f /container-data/data/ssh_pwd ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i " + os.path.join(os.path.abspath(os.path.join(self._abs_root, "..")), "keys/", self._config['inhouse']['machine-name'], "id_rsa") + " " + self._remote_user + "@" + self._config["inhouse"]["address"] + " " + '\"'+ remote_command + '\"'
        logging.info("ssh command is %s", ssh_command)
        cmd = subprocess.Popen(ssh_command, shell=True)
        retcode = cmd.wait()
        if retcode == 0:
            logging.info("SSH completed")
        else:
            logging.error("Error in SSH")
            sys.exit(1)

def parse_args(parser):
    """
    Add to the command line parser the options related to the remote server (none so far)
    """
    return

