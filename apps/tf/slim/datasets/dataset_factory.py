# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A factory-pattern class which returns classification image/label pairs.

Currently not all the dataset present in this factory are implemented, but only flowers, vqa_dataset and imagenet
Due to execution of experiments, I have created 2 imagenet dataset.

Imagenet create the datasets starting from photos in the training directory, instead, Imagenet_records create the dataset
from TFRecords file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import flowers
from datasets import imagenet
from datasets import vqa_dataset
from datasets import imagenet_records
from datasets import vqa_dataset_records

datasets_map = {
    'flowers': flowers,
    'imagenet': imagenet,
    'imagenet_records': imagenet_records,
    'vqa' : vqa_dataset,
    'vqa_records': vqa_dataset_records
}


def get_dataset(name, split_name, dataset_dir, seed, batch_size, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(
      split_name,
      dataset_dir,
      seed,
      batch_size,
      file_pattern,
      reader)
