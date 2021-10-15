"""
Copyright 2019 Marco Lattuada

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

import xmltodict

def load_xml_configuration(parameters, application, root_tag):
    """
    Create a configuration dictionary combining comma-seperated list of parameters and xml file"

    Parameters
    ----------
    parameters: str
        A comma-separated list of parameters in the form parameter=value

    application: str
        The name of the application

    root_tag: str
        The name of the root tag of the xml file.

    Return
    ------
    dict of str: dict of str: str
        A dictionary containing the combination of input parameters and default configuration file
    """
    configuration_base = "default"
    #First look for configuration
    for parameter in parameters.split(","):
        if len(parameter.split("=")) != 2:
            logging.error("parameters must be a , seperated list of <parameter>=<value>: %s", parameter)
            sys.exit(1)
        if parameter.split("=")[0] == "configuration":
            configuration_base = parameter.split("=")[1]
            break

    utility = __import__("utility")
    root_project = utility.get_project_root()

    #The absolute path of the configuration directory
    confs_dir = os.path.join(root_project, "apps", application, "confs")
    logging.info("conf directory is %s", confs_dir)

    #Check the confs_dir exists
    if not os.path.exists(confs_dir):
        logging.error("Conf directory %s does not exist", confs_dir)
        sys.exit(1)

    #Check if xml file of the conf exist
    xml_file_name = os.path.join(confs_dir, configuration_base + ".xml")
    if not os.path.exists(xml_file_name):
        logging.error("XML file %s not found", xml_file_name)
        sys.exit(1)


    #Load XML file
    with open(xml_file_name) as xml_file:
        doc = xmltodict.parse(xml_file.read(), force_list={'input_class'})
    return doc[root_tag]
