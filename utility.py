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

def get_project_root():
    """
    Return
    ------
    str
        The absolute path of the library
    """
    #The absolute path of the current script
    abs_script = os.path.realpath(__file__)

    #The root direcotry of the script
    abs_root = os.path.dirname(abs_script)

    return abs_root

def get_application_module(application):
    """
    Return the python module of a wrapped application

    Parameters
    ----------
    application: str
        The name of the application

    Return
    ------
    The python module containing the wrapper to the application
    """
    project_root = get_project_root()

    #The absolute path of the applications directory
    apps_dir = os.path.join(project_root, "apps")
    logging.info("apps directory is %s", apps_dir)

    #Check the apps_dir exists
    if not os.path.exists(apps_dir):
        logging.error("Apps directory %s does not exist", apps_dir)
        sys.exit(1)

    #Look for app
    if not os.path.exists(os.path.join(apps_dir, application + ".py")):
        logging.error("App %s  not found in %s", application, apps_dir)
        sys.exit(1)

    #Add app to include dir of python
    sys.path.append(apps_dir)

    #Import application
    app_module = __import__(application)
    return app_module

