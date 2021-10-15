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

class Provider:
    """
    Base class for all providers (classes to provide access to different type of targets)

    Attributes
    ----------
    _config: dict of str: dict of str: sttr
        The dictionary generated from the configuration file

    _args
        The parsed command line arguments

    _list_file_name: str
        The file containing the list of experiments to be performed

    _remote_user: str
        The username to be used on the target machine
    """
    _config = None

    _args = None

    _list_file_name = None

    _remote_user = None

    def __init__(self, config, args):
        """
        Arguments
        ---------
        config: dict of str: dict of str: str
            The dictionary created from configuration file

        args
            The command line arguments parsed
        """
        self._config = config
        self._args = args
