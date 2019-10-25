#!/usr/bin/env python3
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
import csv
import matplotlib
#Disable visualization
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

#Parsing input arguments
parser = argparse.ArgumentParser(description="Train an alexnet network by means pytorch")
parser.add_argument("filenames", help="The csv files to be used", nargs='*', action="append")
parser.add_argument('-o', "--output", help="The name of the produced file", required=True)
parser.add_argument('-a', "--aggregate", help="The number of iterations to be aggregated", default=100)
parser.add_argument('-s', "--source", help="The directory storing the logs")
args = parser.parse_args()
csv_data = dict()
data_data = dict()
training_data = dict()
test_data = dict()
x_size = 0
indices = dict()
index = 0

for filename in args.filenames[0]:
   key = os.path.splitext(os.path.basename(filename))[0]
   csv_data[key] = csv.reader(open(filename, "r"), delimiter = ",")
   indices[key] = index
   index = index + 1

#For every file
for key in csv_data:
   data_data[key] = []
   test_data[key] = []
   training_data[key] = []
   iteration_count = 0
   partial_data_sum = 0
   partial_training_sum = 0
   partial_testing_sum = 0

   #Skip header
   next(csv_data[key], None)
   #Add data to y
   for row in csv_data[key]:
      epoch_number = int(row[0])
      type_t = row[1]
      try:
          iteration_number = int(row[2])
      except ValueError:
          print(row)
          raise
      #If we have reached iteration count or we are in the first iteration of the epoch, but not in the first iteration of the first epoch
      if (iteration_number == 0 and (epoch_number != 0 or type_t != "Training")) or iteration_count == args.aggregate:
         average_data = float(partial_data_sum)/float(iteration_count)
         average_training = float(partial_training_sum)/float(iteration_count)
         average_testing = float(partial_testing_sum)/float(iteration_count)
         #We are printing value until the last iteration, so the type is the opposite
         if (type_t.find("Training") != -1 and iteration_number != 0) or (type_t.find("Testing") != -1 and iteration_number == 0):
            for i in range(0, iteration_count):
               data_data[key].append(average_data)
               training_data[key].append(average_training)
               test_data[key].append(0.0)
         else:
            for i in range(0, iteration_count):
               data_data[key].append(0.0)
               training_data[key].append(0.0)
               test_data[key].append(average_testing)
         iteration_count = 0
         partial_data_sum = 0.0
         partial_training_sum = 0.0
         partial_testing_sum = 0.0
      partial_data_sum = partial_data_sum + float(row[3])
      partial_training_sum = partial_training_sum + float(row[4])
      partial_testing_sum = partial_testing_sum + float(row[5])
      iteration_count = iteration_count + 1
   #Update max value of x
   if len(training_data[key]) > x_size:
      x_size = len(training_data[key])

x_values = list(range(0, x_size))


for key in csv_data:
   plt.plot(x_values, data_data[key], label = "loading_" + str(indices[key]))
   plt.plot(x_values, training_data[key], label = "training_" + str(indices[key]))
   plt.plot(x_values, test_data[key], label = "test_" + str(indices[key]))
title = ""
for filename in args.filenames[0]:
   if title != "":
      title = title + "\n"
   if args.source != None:
      title = args.source + "\n"
   tokens = filename.split("_")
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
   title = title + title_row
plt.title(title, fontsize=5)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(args.output)
