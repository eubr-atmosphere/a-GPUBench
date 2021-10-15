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
"""Provides data for the ImageNet ILSVRC 2012 Dataset plus some bounding boxes.

Some images have one or more bounding boxes associated with the label of the
image. See details here: http://image-net.org/download-bboxes

ImageNet is based upon WordNet 3.0. To uniquely identify a synset, we use
"WordNet ID" (wnid), which is a concatenation of POS ( i.e. part of speech )
and SYNSET OFFSET of WordNet. For more information, please refer to the
WordNet documentation[http://wordnet.princeton.edu/wordnet/documentation/].

"There are bounding boxes for over 3000 popular synsets available.
For each synset, there are on average 150 images with bounding boxes."

WARNING: Don't use for object detection, in this case all the bounding boxes
of the image belong to just one class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tensorflow as tf
from PIL import Image
import numpy as np 

from datasets import dataset_utils

# TODO(nsilberman): Add tfrecord file type once the script is updated.
_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {
    'train': 12000,
    'validation': 500,
}

_NUM_SAMPLES = _SPLITS_TO_SIZES['train'] + _SPLITS_TO_SIZES['validation']

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each object.',
}

_NUM_CLASSES = 10

# If set to false, will not try to set label_to_names in dataset
# by reading them from labels.txt or github.
LOAD_READABLE_NAMES = True


def create_readable_names_for_imagenet_labels():
  """Create a dict mapping label id to human readable string.

  Returns:
      labels_to_names: dictionary where keys are integers from to 1000
      and values are human-readable names.

  We retrieve a synset file, which contains a list of valid synset labels used
  by ILSVRC competition. There is one synset one per line, eg.
          #   n01440764
          #   n01443537
  We also retrieve a synset_to_human_file, which contains a mapping from synsets
  to human-readable names for every synset in Imagenet. These are stored in a
  tsv format, as follows:
          #   n02119247    black fox
          #   n02119359    silver fox
  We assign each synset (in alphabetical order) an integer, starting from 1
  (since 0 is reserved for the background class).

  Code is based on
  https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py#L463
  """

  # pylint: disable=g-line-too-long
  base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/'
  synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
  synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url)

  filename, _ = urllib.request.urlretrinpeve(synset_url)
  synset_list = [s.strip() for s in open(filename).readlines()]
  num_synsets_in_ilsvrc = len(synset_list)
  assert num_synsets_in_ilsvrc == 1000

  filename, _ = urllib.request.urlretrieve(synset_to_human_url)
  synset_to_human_list = open(filename).readlines()
  num_synsets_in_all_imagenet = len(synset_to_human_list)
  assert num_synsets_in_all_imagenet == 21842

  synset_to_human = {}
  for s in synset_to_human_list:
    parts = s.strip().split('\t')
    assert len(parts) == 2
    synset = parts[0]
    human = parts[1]
    synset_to_human[synset] = human

  label_index = 1
  labels_to_names = {0: 'background'}
  for synset in synset_list:
    name = synset_to_human[synset]
    labels_to_names[label_index] = name
    label_index += 1

  return labels_to_names

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

def get_split(split_name, dataset_dir, seed, batch_size, file_pattern=None, reader=None,):
  """Gets a dataset tuple with instructions for reading ImageNet.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  filenames = os.listdir(dataset_dir)
  filenames_ = []
  for f in filenames:
    filenames_.append(dataset_dir+"/"+f)
  
  
  dataset = tf.data.TFRecordDataset(filenames_)


  keys_to_features = {
      'image/encoded':  tf.io.FixedLenFeature([], tf.string),
      'image/format':  tf.io.FixedLenFeature([], tf.string, default_value='jpg'),
      'image/class/label':  tf.io.FixedLenFeature([], tf.int64, default_value=-1),
      'image/class/text':  tf.io.FixedLenFeature([], tf.string, default_value=''),
      'image/object/bbox/xmin':  tf.io.VarLenFeature(tf.float32),
      'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
      'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
      'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
      'image/object/class/label': tf.io.VarLenFeature(tf.int64),
  }
  def decode_image(image):
    #image = image_features['image/encoded']
    img = tf.io.decode_image(image, channels=3)
    #image = tf.io.decode_raw(image,tf.float32)
    #image = tf.cast(image, tf.float32)
    #image = tf.reshape(image, tf.stack([*image_size, 3]))
    image = tf.image.resize_with_pad(img, 224, 224, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image

  def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
   example = tf.io.parse_single_example(example_proto, keys_to_features)
  #exit(tf.cast(example['image/width'], tf.float32))
   image = decode_image(example['image/encoded'])

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


  train_dataset, validation_dataset = split_dataset(parsed_image_dataset, _SPLITS_TO_SIZES['validation']/_SPLITS_TO_SIZES['train']  )
  
  train_dataset = train_dataset.batch(batch_size) 

  train_dataset = train_dataset.repeat()  
  
  #if(args.dataset_name == "vqa"):
  validation_dataset = validation_dataset.batch(batch_size)
   
  validation_dataset = validation_dataset.repeat()



  #parsed_image_dataset = parsed_image_dataset.repeat()

  #parsed_image_dataset

  #for class_id in parsed_image_dataset.take(10):
  # print(class_id['image/width'])
  # image = tf.image.decode_jpeg(class_id['image/encoded'], channels=3)
  # image = tf.cast(image, tf.float32)
  # image = tf.reshape(image, [class_id['image/width'],class_id['image/height'], 3])
  # images.append(tf.random.uniform(shape=[32, 224,224,3]))

   
  #image, label = next(iter(parsed_image_dataset))
  #print(image)
  #print(label)

  #img = Image.fromarray(np.uint8(image))
  #img = img.resize([224,224])

  #img.show()

  #exit()

  return _NUM_CLASSES, _SPLITS_TO_SIZES['train'],_SPLITS_TO_SIZES['validation'], train_dataset, validation_dataset

#Returns the size of training set and validation set
def get_split_size():
    return SPLITS_TO_SIZES['train'],SPLITS_TO_SIZES['validation']
  #labels_to_names = None
  #if LOAD_READABLE_NAMES:
  #  if dataset_utils.has_labels(dataset_dir):
  #    labels_to_names = dataset_utils.read_label_file(dataset_dir)
  #  else:
  #    labels_to_names = create_readable_names_for_imagenet_labels()
  #    dataset_utils.write_label_file(labels_to_names, dataset_dir)

