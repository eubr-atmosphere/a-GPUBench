# a-GPUBench

Framework composed of a collection of python script to run, profile and collect data about applications exploiting GPUs. Application runs can be started by means of launch_experiment.py script.

The framework can be used with different machines and different applicaitons.
The target architecture already supported by this version are:
- inhouse server
- local machine

The application already supported by this version are:
- CNN training with pytorch
- CNN and RNN training with tensorflow

The framework can be configured via .ini configuration file.
An example of configuration file is available in configurations/default.ini.

Support to new providers can be provided by adding a python package under providers.
The package must provide the following functions:
- copy_list_to_target: to copy the list of experiments to be run from localhost to target
- initialize: to initialize the target architecture
- parse_args: to add command line arguments specific of the target architecture
- run_experiment: to run the experiment(s)

Support to new applications (not limited to python implementations) can be provided by adding a python package under apps which wrap them.
The package must provide the following functions:
- compute_configuration_name: to compute the name of the configuration of an experiment
- collect_data: to parse the output of an experiment and generate results
- main: to execute experiement(s)

The code in this repository is licensed under the terms of the
[Apache License version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
