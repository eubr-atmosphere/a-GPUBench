#!/usr/bin/python3
"""
Copyright 2018-2019 Marco Lattuada

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
import shutil
import sys
import tempfile

def main():
    """
    Main script of the library

    This script must be invoked after preparing the list of experiments to be performed (and the configuration file if necessary)

    The arguments of the script are:
    -c, --configuration-file (optional): the configuration file containing information about the targets. If this option is not used, the "polimi.ini" file will be used
    -a, --application (mandatory): the application to be profiled. The library looks for a wrapper with the same name under apps directory
    -p, --parameters (mandatory): it can be the parameters (comma separated list) of a single experiments or the name of the file containing the list of experiments to be performed
    -r, --repetitions (optional): how many times each experiment must be re-execute; the default value is 1
    -d, --debug (optional): enables the debug messages; the default value is False
    -o, --output (optional): where the output information will be stored (on the machine actually running the experiment)
    --provider (mandatory): the type of target to be used; the library looks for a python file with the same name under providers; at the moment the supported targets are: local (the same machine running this script), inhouse (a server which can be accessed through ssh), azure (a Microsoft Azure VM)
    --mail: the mail address to which a message is sent at the end of each experiment; the mail is sent by the target server
    --profile: enables the CPU and GPU profiling during experiment execution
    """

    parser = argparse.ArgumentParser(description="Execute experiment on AZ machine")

    parser.add_argument('-c', "--configuration-file", help="The configuration file for the infrastructure", default=os.path.join("configurations", "polimi.ini"))
    parser.add_argument('-a', "--application", help="The application to be run. It must correspond to an execuatble or a script present in directory apps", required=True)
    parser.add_argument('-p', "--parameters", help="The parameters of the experiment", required=True)
    parser.add_argument('-r', "--repetitions", help="The number of times the application has to be executed", default=1)
    parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    parser.add_argument('-o', "--output", help="The output root directory", default="output")
    parser.add_argument('--provider', help="The type of remote machine to be used", required=True)
    parser.add_argument("--mail", help="The mail address to which end notification must be sent")
    parser.add_argument("--profile", help="Log resource usage", default="")

    #The absolute path of the current script
    abs_script = os.path.abspath(sys.argv[0])

    #The root directory of the script
    abs_root = os.path.dirname(abs_script)

    #The providers directory
    providers_dir = os.path.join(abs_root, "providers")

    #Add providers directory to python paths
    sys.path.append(providers_dir)


    #Register providers
    providers_modules = {}
    for provider in os.listdir(providers_dir):
        provider_path = os.path.join(providers_dir, provider)
        #Get only python files
        if os.path.isfile(provider_path) and provider.endswith(".py") and provider != "provider.py":
            provider_name = os.path.splitext(provider)[0]
            providers_modules[provider_name] = __import__(provider_name)
            providers_modules[provider_name].parse_args(parser)

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    current_configuration_file_name = os.path.join(abs_root, "configurations", "current.ini")
    shutil.copyfile(args.configuration_file, current_configuration_file_name)

    #Read configuration file
    config = configparser.ConfigParser()
    config.read(current_configuration_file_name)

    if args.provider not in providers_modules:
        logging.error("Provider %s not available", args.provider)
        sys.exit(1)

    provider_module = providers_modules[args.provider]


    #Check for mail
    if config["global"]["must_have_email"] and config["global"]["must_have_email"] == "True" and not args.mail:
        logging.error("--mail option set as mandatory")
        sys.exit(1)

    #Initialize provider wrapper
    logging.info("Initialize provider wrapper")
    provider = provider_module.Provider(config, args)

    #Check that keys are available
    if not os.path.exists(os.path.join(abs_root, "keys", "id_rsa")) or not os.path.exists(os.path.join(abs_root, "keys", "id_rsa.pub")):
        logging.error("Please add private/public keys in keys directory")
        sys.exit(1)

    #If parameters is not a file create a file with a single line
    if "=" in args.parameters:
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        temp_file.write(args.parameters)
        temp_file.close()
        list_file_name = temp_file.name
    elif os.path.exists(args.parameters):
        list_file_name = os.path.abspath(args.parameters)
    else:
        logging.error("Parameters do not describe a configuration nor a file")
        sys.exit(1)

    #Copy the list to VM
    logging.info("Copying list file %s to target", list_file_name)
    provider.copy_list_to_target(list_file_name)

    #Run the experiment
    provider.run_experiment()

    sys.exit(0)

if __name__ == "__main__":
    main()
