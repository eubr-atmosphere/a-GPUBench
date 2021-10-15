# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright 2021 Giovanni Dispoto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers

$ python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python download_and_convert_data.py \
    --dataset_name=visualwakewords \
    --dataset_dir=/tmp/visualwakewords

```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse

from datasets import download_and_convert_flowers
from datasets import download_and_convert_mnist
from datasets import download_and_convert_vqa

parser = argparse.ArgumentParser()


parser.add_argument(
    '--dataset_name',
    default = '',
    help = 'The name of the dataset to convert, one of "flowers" or "VQA"'
    )

parser.add_argument(
    '--dataset_dir',
    default = '',
    help = 'The directory where the output TFRecords and temporary files are saved.')

parser.add_argument(
    '--small_object_area_threshold', default  = 0.005, type=int,
    help = 'For --dataset_name=visualwakewords only. Threshold of fraction of image area below which small objects are filtered')

parser.add_argument(
    '--foreground_class_of_interest', default =  'person',
    help = 'For --dataset_name=visualwakewords only. Build a binary classifier based on the presence or absence of this object in the image.')

args = parser.parse_args()

def main():
  if args.dataset_name == '':
    raise ValueError('You must supply the dataset name with --dataset_name')
  if args.dataset_dir == '':
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if args.dataset_name == 'flowers':
    download_and_convert_flowers.run(args.dataset_dir)
  elif args.dataset_name == 'vqa':
    download_and_convert_vqa.run(args.dataset_dir)  
  else:
    raise ValueError(
        'dataset_name [%s] was not recognized.' % args.dataset_name)

if __name__ == '__main__':
  main()
