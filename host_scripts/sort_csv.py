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
import csvsort
import pandas
import shutil

parser = argparse.ArgumentParser(description="Sort cvs by columns")
parser.add_argument("-i", "--input", help="The input file", required=True)
parser.add_argument("-o", "--output", help="The output file", required=True)
parser.add_argument("-c", "--columns", help="The list of columns to be considered", required=True)
args = parser.parse_args()

#For each column index, the full column name
full_column_names = {}

data = pandas.read_csv(args.input, na_values="-")
index = 0
for column_name in data.columns:
   full_column_names[index] = column_name
   index = index + 1

column_names = []
for column in args.columns.split(","):
    column_names.append(full_column_names[int(column)])

sorted_data = data.sort_values(by=column_names)
sorted_data.to_csv(args.output, index=False, na_rep="NaN")
