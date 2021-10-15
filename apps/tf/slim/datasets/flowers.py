# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright 2021 Giovanni Dispoto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets import dataset_utils
from PIL import Image
import numpy as np

_FILE_PATTERN = 'flowers_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 3320, 'validation': 350}

_NUM_CLASSES = 5

_NUM_SAMPLES = SPLITS_TO_SIZES['train'] + SPLITS_TO_SIZES['validation']

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}

def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    rounded up to two decimal places.
    @param dataset: the input dataset to split.
    @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
    @return: a tuple of two tf.data.Datasets as (training, validation)
    """

    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset


def get_split(split_name, dataset_dir, seed, batch_size,file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """

  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  #f reader is None:
  #  reader = tf.TFRecordReader

  #keys_to_features = {
  #    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
  #    'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
  #    'image/class/label': tf.FixedLenFeature(
  #        [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  #}

  #items_to_handlers = {
  #    'image': slim.tfexample_decoder.Image(),
  #    'label': slim.tfexample_decoder.Tensor('image/class/label'),
  #}

  #decoder = slim.tfexample_decoder.TFExampleDecoder(
  #    keys_to_features, items_to_handlers)

  #labels_to_names = None
  #if dataset_utils.has_labels(dataset_dir):
  #  labels_to_names = dataset_utils.read_label_file(dataset_dir)

  filenames = os.listdir(dataset_dir)
  filenames_ = []
  for f in filenames:
    filenames_.append(dataset_dir+"/"+f)
  
  
  dataset = tf.data.TFRecordDataset(filenames_)



  image_feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value='png'),
    'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
  }
  def decode_image(image, image_size):
    #image = image_features['image/encoded']
    img = tf.io.decode_image(image, channels=3)
    #image = tf.io.decode_raw(image,tf.float32)
    #image = tf.cast(image, tf.float32)
    #image = tf.reshape(image, tf.stack([*image_size, 3]))
    image = tf.image.resize_with_pad(img, 224, 224, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image

  def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
   example = tf.io.parse_single_example(example_proto, image_feature_description)
  #exit(tf.cast(example['image/width'], tf.float32))
   image = decode_image(example['image/encoded'], [tf.cast(example['image/height'], tf.int32), tf.cast(example['image/width'], tf.int32)])

   return image, tf.one_hot(tf.cast(example['image/class/label'], tf.int64), depth=_NUM_CLASSES)

  parsed_image_dataset = dataset.map(_parse_image_function)

  #for image_features in parsed_image_dataset:
  # image = image_features['image/encoded']
  # img = tf.io.decode_image(image, channels=3)
  # sample_ = img[0, ...]

  # img = Image.fromarray(np.uint8(np.array(img)))
  # img = img.resize([224,224])

  # img.show()
  # print(image_features['image/class/label'])
  # exit(img)


  parsed_image_dataset = parsed_image_dataset.shuffle(2048, seed = seed)

  parsed_image_dataset = parsed_image_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)


  train_dataset, validation_dataset = split_dataset(parsed_image_dataset, SPLITS_TO_SIZES['validation']/SPLITS_TO_SIZES['train']  )
  


  #parsed_image_dataset = parsed_image_dataset.repeat()

  #parsed_image_dataset

  #for class_id in parsed_image_dataset.take(10):
  # print(class_id['image/width'])
  # image = tf.image.decode_jpeg(class_id['image/encoded'], channels=3)
  # image = tf.cast(image, tf.float32)
  # image = tf.reshape(image, [class_id['image/width'],class_id['image/height'], 3])
  # images.append(tf.random.uniform(shape=[32, 224,224,3]))

   
  #image, label = next(iter(train_dataset))
  #print(image)
  #print(label)

  #img = Image.fromarray(np.uint8(image))
  #img = img.resize([224,224])

  #img.show()

  train_dataset = train_dataset.batch(batch_size) 

  train_dataset = train_dataset.repeat()  
  
  #if(args.dataset_name == "vqa"):
  validation_dataset = validation_dataset.batch(batch_size)
   
  validation_dataset = validation_dataset.repeat()

  return _NUM_CLASSES, SPLITS_TO_SIZES['train'],SPLITS_TO_SIZES['validation'], train_dataset, validation_dataset


#Returns the size of training set and validation set
def get_split_size():
    return SPLITS_TO_SIZES['train'],SPLITS_TO_SIZES['validation']

  #return slim.dataset.Dataset(
  #    data_sources=file_pattern,
  #    reader=reader,
  #    decoder=decoder,
  #    num_samples=SPLITS_TO_SIZES[split_name],
  #    items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
  #    num_classes=_NUM_CLASSES,
  #
  #     labels_to_names=labels_to_names)
  