#!/usr/bin/python3
"""
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
import xmltodict
import os
import logging
import sys
import subprocess

#Elements used in grid search
network_type = ['vgg_16', 'resnet_v1_50', 'alexnet_v2']
network_depth = [1,2,3] #for resnet
optimizers = ['adam', 'momentum', 'adadelta']
momentum = [0.9, 0.8, 0.7]
learning_rate_decay_type = ['fixed', 'polynomial', 'exponential']
learning_rate = [0.001, 0.0001, 0.00001]
dropout = [0.2, 0.3, 0.4]
batch_size = [32,64]

#base xml used in order to fill it with information and run a training session
config_name = 'base_grid.xml'

for network in network_type:
    for optimizer in optimizers:
            for lr in learning_rate:
                for drop in dropout:
                    for bs in batch_size:
                        xml_file = os.path.join("/data/src/apps/tf/confs", config_name)
                        if not os.path.exists(xml_file):
                            logging.error("XML file %s not found", xml_file)
                            sys.exit(1)

                        # Load XML file
                        with open(xml_file) as fd:
                         doc = xmltodict.parse(fd.read(), force_list={'input_class'})

                        #fill xml with information obtained from grid search
                        #try with one configuration
                        doc['tensorflow_configuration']['network_type'] = network
                        doc['tensorflow_configuration']['optimizer'] = optimizer
                        doc['tensorflow_configuration']['momentum'] = 0.7 #not always used, but setted anyway
                        doc['tensorflow_configuration']['learning_rate'] = lr
                        doc['tensorflow_configuration']['dropout'] = drop
                        doc['tensorflow_configuration']['batch_size'] = bs
                        doc['tensorflow_configuration']['learning_rate_decay_type'] = 'fixed' #using fixed learning rate

                        #save configuration
                        with open(os.path.join(base_path, 'test_1.xml'), 'w') as result_file:
                         result_file.write(xmltodict.unparse(doc))

                        #modify grid_conf file specify which configuration to use
                        f = open(os.path.join(base_path,'grid_conf'), "w")
                        f.write("configuration="+"test_1")
                        f.close()

                        #try a run
                        command = "python3 ./app/vm_scripts/launch_local_experiment.py -a tf --parameters-list /data/config/grid_conf  --profile GPU --output ./data/output"
                        logging.info("command is %s", command)
                            
                        cmd = subprocess.Popen(command, shell=True)
                        retcode = cmd.wait()
                        if retcode == 0:
                         logging.info("launched experiment")
                        else:
                         logging.error("Error in launching  experiment")
                         sys.exit(1)