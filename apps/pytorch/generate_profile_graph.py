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
import linecache
import logging
import matplotlib
#Disable visualization
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
import sys
import traceback

#Parsing input arguments
parser = argparse.ArgumentParser(description="Train an alexnet network by means pytorch")
parser.add_argument('-c', "--cpu-profile", help="The file storing the CPU(s) profile usage", required=True)
parser.add_argument('-g', "--gpu-profile", help="The file storing the GPU(s) profile usage", required=True)
parser.add_argument('-o', "--output", help="The name of the produced file", required=True)
parser.add_argument('-s', "--source", help="The directory storing the logs")
parser.add_argument('-i', "--iteration-file", help="The file containing information about iteration")
parser.add_argument('-t', "--overall-time", help="The overall execution time")
parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
args = parser.parse_args()

scale = 1000000000

#Initializing logger
if args.debug:
   logging.basicConfig(level=logging.DEBUG,format='%(levelname)s: %(message)s')
else:
   logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
try:
   #Open csv file
   csv_cpu_file = open(args.cpu_profile, "r")
   csv_gpu_file = open(args.gpu_profile, "r")

   cpu_row_0 = linecache.getline(args.cpu_profile, 1)
   gpu_row_0 = linecache.getline(args.gpu_profile, 1)

   cpu_row_1 = linecache.getline(args.cpu_profile, 2)
   gpu_row_1 = linecache.getline(args.gpu_profile, 2)
   #Get the number of CPUs and GPUs
   cpu_number = len(cpu_row_0.split(","))-2
   gpu_number = int((len(gpu_row_0.split(","))-2)/2)

   #Get the initial time
   try:
      cpu_initial_time = int(cpu_row_1.split(",")[0])
   except:
      print("Error in looking for initial CPU time parsing " + cpu_row_1)
      raise
   gpu_initial_time = int(gpu_row_1.split(",")[0])
   overall_initial_time = min(cpu_initial_time, gpu_initial_time)

   cpu_x = []
   cpu_y = []
   gpu_x = []
   gpu_y = []
   gpu_memory_y = []


   #Add CPU data
   for cpu_row in csv_cpu_file:
      split = cpu_row.split(",")
      if split[0] == "timestamp":
         continue
      cpu_x.append((int(split[0]) - overall_initial_time)/scale)
      cpu_usage = 0
      for cpu_index in range(0, cpu_number):
         cpu_usage = cpu_usage + float(split[cpu_index + 2])
      cpu_y.append(cpu_usage)

   index = 1
   #Add GPU data
   for gpu_row in csv_gpu_file:
      split = gpu_row.split(",")
      if split[0] == "timestamp":
         continue
      gpu_x.append((int(split[0]) - overall_initial_time)/scale)
      gpu_usage = 0
      gpu_memory_usage = 0
      for gpu_index in range(0, gpu_number):
         gpu_usage = gpu_usage + float(split[gpu_index + 2])
         gpu_memory_usage = gpu_memory_usage + float(split[gpu_index + gpu_number + 2])
      gpu_y.append(gpu_usage)
      gpu_memory_y.append(gpu_memory_usage)

   plt.plot(cpu_x, cpu_y, label = "CPU usage")
   plt.plot(gpu_x, gpu_y, label = "GPU usage")
   plt.plot(gpu_x, gpu_memory_y, label = "GPU memory usage")
   plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
   tokens = os.path.basename(args.cpu_profile).split("_")
   title_row = "GPU: " + tokens[2]
   title_row = title_row + " * Number: " + tokens[3]
   title_row = title_row + " * Network: " + tokens[4]
   for token_index in range(5, len(tokens)-3):
      if tokens[token_index] == "cl":
         title_row = title_row + " * Classes:"
      elif tokens[token_index] == "im":
         title_row = title_row + " * Images:"
      elif tokens[token_index] == "ep":
         title_row = title_row + " * Epochs:"
      elif tokens[token_index] == "bs":
         title_row = title_row + " * Batch Size:"
      elif tokens[token_index] == "mo":
         title_row = title_row + " * Momentum:"
      elif tokens[token_index] == "profile":
         title_row = title_row + " * Profile:"
      elif tokens[token_index] == "j":
         title_row = title_row + " * Threads:"
      elif tokens[token_index - 1] == "im" and tokens[token_index] == "9999":
         title_row = title_row + " MAX"
      else:
          title_row = title_row + " " + tokens[token_index]
   if args.source != None:
      title = args.source + "\n"
   else:
      title = ""
   title = title + title_row
   if args.overall_time != None:
      title = title + "\nOverall execution time: " + args.overall_time
   plt.title(title, fontsize=5)
   if args.iteration_file != None:
      logging.debug("iteration file is " + args.iteration_file)
      iteration_file = open(args.iteration_file, "r")
      end_time = ""
      for iteration_row in iteration_file:
         split = iteration_row.split(",")
         if split[0] == "Epoch":
            continue
         if end_time != "" and split[2] == "0":
            if split[1] == "Training":
               end_time_string = (float(split[6].replace('\n', ""))*1000000000 - float(split[3]) - float(split[4]) - overall_initial_time)/scale
            else:
               end_time_string = (float(end_time)*1000000000 - overall_initial_time)/scale
            plt.axvline(x=end_time_string,color='r')
         end_time = split[6].replace('\n', "")
   plt.xlabel("Time [s]")
   plt.ylabel("Usage [%]")

   plt.gcf().subplots_adjust(bottom=0.15)
   plt.savefig(args.output)
except:
   print("Error in generation of profile " + args.output)
   print("Input is " + args.cpu_profile)
   print("Input is " + args.gpu_profile)
   traceback.print_exc()
   sys.exit(-1)
