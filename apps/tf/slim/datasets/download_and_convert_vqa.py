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
r"""
This script create TFRecords from VQA dataset.
In order to work with downloaded dataset, change the DATA_URL with URL of dataset.
If you want to work with local dataset, write only the name of the dataset in URL and run the script.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import pandas as pd
import numpy as np

from six.moves import range
from six.moves import zip
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'anndl-2020-vqa.zip'

# The number of images in the validation set.
_NUM_VALIDATION = 8823


# Seed for repeatability.
_RANDOM_SEED = 0

# Maximum question len, used for padding
_MAX_QUESTION_LEN = 21

# The number of shards per dataset split.
_NUM_SHARDS = 5

labels_dict = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        'apple': 6,
        'baseball': 7,
        'bench': 8,
        'bike': 9,
        'bird': 10,
        'black': 11,
        'blanket': 12,
        'blue': 13,
        'bone': 14,
        'book': 15,
        'boy': 16,
        'brown': 17,
        'cat': 18,
        'chair': 19,
        'couch': 20,
        'dog': 21,
        'floor': 22,
        'food': 23,
        'football': 24,
        'girl': 25,
        'grass': 26,
        'gray': 27,
        'green': 28,
        'left': 29,
        'log': 30,
        'man': 31,
        'monkey bars': 32,
        'no': 33,
        'nothing': 34,
        'orange': 35,
        'pie': 36,
        'plant': 37,
        'playing': 38,
        'red': 39,
        'right': 40,
        'rug': 41,
        'sandbox': 42,
        'sitting': 43,
        'sleeping': 44,
        'soccer': 45,
        'squirrel': 46,
        'standing': 47,
        'stool': 48,
        'sunny': 49,
        'table': 50,
        'tree': 51,
        'watermelon': 52,
        'white': 53,
        'wine': 54,
        'woman': 55,
        'yellow': 56,
        'yes': 57
}


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
   # self._decode_jpeg_data =  #tf.placeholder(dtype=tf.string)
   # self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
   print("init")

  def read_image_dims(self, image_data):
    image = self.decode_jpeg(image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self,image_data):
    image = tf.image.decode_jpeg(image_data, channels=3) #sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
    # assert len(image.shape) == 3
    # assert image.shape[2] == 3
    # return image
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred question and answers.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
   A list of question, related image in the directory and answers.
  """
  train_questions = pd.read_json(os.path.join(dataset_dir, 'VQA_Dataset/train_questions_annotations.json'))
  #Questions
  questions = list(train_questions.iloc[0])
  #Answers
  answers = list(train_questions.iloc[2])
  #Images
  images = np.array(train_questions.iloc[1])
  images_path = []
  
  questions_tokenizer = Tokenizer()
  questions_tokenizer.fit_on_texts(questions)
  questions_tokenized = questions_tokenizer.texts_to_sequences(questions)

  questions_encoder_inputs = pad_sequences(questions_tokenized, maxlen=_MAX_QUESTION_LEN)

  for image in images:
    path = os.path.join(os.path.join(dataset_dir, 'VQA_Dataset/Images'), image+".png")
    images_path.append(path)

  return questions_encoder_inputs, images_path, answers


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'vqa_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, questions, answers, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    questions : A list of questions associated to files
    answers: A list of answers associated to files
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))


  image_reader = ImageReader()

    #with tf.Session('') as sess:

  for shard_id in range(_NUM_SHARDS):
    output_filename = _get_dataset_filename(
    dataset_dir, split_name, shard_id)

    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_ndx = shard_id * num_per_shard
      end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
      for i in range(start_ndx, end_ndx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
        sys.stdout.flush()

        # Read the filename:
        image_data = tf.io.gfile.GFile(filenames[i], 'rb').read()
        height, width = image_reader.read_image_dims(image_data)

        #class_name = os.path.basename(os.path.dirname(filenames[i]))
        question = questions[i]
        answer = labels_dict[answers[i]]

        example = dataset_utils.image_to_tfexample_vqa(
        image_data, b'png', height, width, question.tolist(), answer)
        tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.io.gfile.remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
  tf.io.gfile.rmtree(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.io.gfile.exists(output_filename):
        return False
  return True


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.io.gfile.exists(dataset_dir):
    tf.io.gfile.mkdir(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  #dataset_utils.download_and_uncompress_zipfile(_DATA_URL, dataset_dir)
  questions, images_path, answers = _get_filenames_and_classes(dataset_dir)

  # Divide into train and test:
  random.seed(_RANDOM_SEED)

  training_filenames = images_path[_NUM_VALIDATION:]
  training_questions = questions[_NUM_VALIDATION:]
  training_answers = answers[_NUM_VALIDATION:]

  validation_filenames = images_path[:_NUM_VALIDATION]
  validation_questions = questions[:_NUM_VALIDATION]
  validation_answers = answers[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, training_questions, training_answers,
                   dataset_dir)

  _convert_dataset('validation', validation_filenames, validation_questions, validation_answers,
                   dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(
      list(zip(labels_dict.values(), labels_dict.keys())))

  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the VQA dataset!')
