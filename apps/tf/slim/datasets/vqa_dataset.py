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
"""Provides data for the VQA dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_vqa.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from datasets import dataset_utils
from PIL import Image
import numpy as np
import pandas as pd


SPLITS_TO_SIZES = {'train': 50007, 'validation': 8823}

_NUM_CLASSES = 58

_FILE_PATTERN = 'vqa_%s_*.tfrecord'

_NUM_SAMPLES = SPLITS_TO_SIZES['train'] + SPLITS_TO_SIZES['validation']

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'question': 'Question related to the image',
    'answer' : 'The answer related to the question'
}

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
  
img_w = 700
img_h = 400
max_questions_length = 21
max_answers_length = 1
num_classes= 58



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


def get_split(split_name, dataset_dir, seed, batch_size, file_pattern=None, reader=None):
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

  train_questions = pd.read_json(os.path.join(dataset_dir, '../raw-data/VQA_Dataset/train_questions_annotations.json'))
  #os.path.join(dataset_dir, '../raw-data/VQA_Dataset/')

  #Questions
  questions = list(train_questions.iloc[0])
  #Answers
  answers = list(train_questions.iloc[2])
  #Images
  images = np.array(train_questions.iloc[1])

  #transform answers into values of dictionary
  for i in range(len(answers)):
   answers[i] = labels_dict[answers[i]]
  

  
  dataset = CustomDataset(os.path.join(dataset_dir, '../raw-data/VQA_Dataset'), 'training', train_questions=train_questions) #preprocessing_function=preprocess_input
  dataset_valid = CustomDataset(os.path.join(dataset_dir, '../raw-data/VQA_Dataset'), 'validation', train_questions=train_questions) #preprocessing_function=preprocess_input

  

  train_dataset = tf.data.Dataset.from_generator(lambda: dataset,
                                               output_types=((tf.int32, tf.float32), tf.int32),
                                               output_shapes=(([max_questions_length], [int(img_h/2), int(img_w/2), 3]), [num_classes]))



  validation_dataset = tf.data.Dataset.from_generator(lambda: dataset_valid,
                                               output_types=(( tf.int32, tf.float32), tf.int32),
                                               output_shapes=(([max_questions_length], [int(img_h/2), int(img_w/2), 3]), [num_classes]))

  #train_dataset = train_dataset.shuffle(2048, seed = seed)

  train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

  #validation_dataset = validation_dataset.shuffle(2048, seed = seed)

  validation_dataset = validation_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

  train_dataset = train_dataset.batch(batch_size) 

  train_dataset = train_dataset.repeat()  
  
  validation_dataset = validation_dataset.batch(batch_size)
   
  validation_dataset = validation_dataset.repeat()


  return _NUM_CLASSES, SPLITS_TO_SIZES['train'] ,SPLITS_TO_SIZES['validation'], train_dataset, validation_dataset


#Returns the size of training set and validation set
def get_split_size():
    return SPLITS_TO_SIZES['train'],SPLITS_TO_SIZES['validation']

  


class CustomDataset(tf.keras.utils.Sequence):

  """
    CustomDataset inheriting from tf.keras.utils.Sequence.

    3 main methods:
      - __init__: save dataset params like directory, filenames..
      - __len__: return the total number of samples in the dataset
      - __getitem__: return a sample from the dataset

    Note: 
      - the custom dataset return a single sample from the dataset. Then, we use 
        a tf.data.Dataset object to group samples into batches.
      - in this case we have a different structure of the dataset in memory. 
        We have all the images in the same folder and the training and validation splits
        are defined in text files.

  """

  def __init__(
    self, dataset_dir,
    which_subset,
    img_generator=None,
    preprocessing_function=None,
    train_questions=None,

  ):

    if which_subset == 'training':
     subset_filenames = list(train_questions.iloc[1])[int(len(train_questions.iloc[1])*0.15)+1:]
     #Questions
     questions = list(train_questions.iloc[0])[int(len(train_questions.iloc[0])*0.15)+1:]
     #Answers
     answers = list(train_questions.iloc[2])[int(len(train_questions.iloc[2])*0.15)+1:]
    elif which_subset == 'validation':
     subset_filenames = list(train_questions.iloc[1])[:int(len(train_questions.iloc[1])*0.15)]
     #Questions
     questions = list(train_questions.iloc[0])[:int(len(train_questions.iloc[0])*0.15)]
     #Answers
     answers = list(train_questions.iloc[2])[:int(len(train_questions.iloc[2])*0.15)]
     #Answers
     #answers = list(test_questions.iloc[2])
  

    # Create Tokenizer to convert words to integers
    questions_tokenizer = Tokenizer()
    questions_tokenizer.fit_on_texts(list(train_questions.iloc[0]))
    questions_tokenized = questions_tokenizer.texts_to_sequences(questions)
    len(questions_tokenized)

    #If dataset is training set or validation set, trasform answer into label
    if(which_subset != 'test'):
      for i in range(len(answers)):
       answers[i] = labels_dict[answers[i]]
    else:
      answers = [] 
    

    questions_encoder_inputs = pad_sequences(questions_tokenized, maxlen=max_questions_length)
   

    self.which_subset = which_subset
    self.dataset_dir = dataset_dir
    self.subset_filenames = subset_filenames
    self.questions = questions_encoder_inputs
    self.answers=answers
    self.img_generator = img_generator
    self.preprocessing_function = preprocessing_function
    self.question_tokenizer = questions_tokenizer

  def __len__(self):
    return len(self.questions)

  def get_questions_tokenizer(self):
    return self.question_tokenizer


  def __getitem__(self, index):
    # Read Image
    curr_filename = self.subset_filenames[index]
    curr_question = (self.questions[index])
    if(self.which_subset != 'test'):
      curr_answer = (self.answers[index])

      zer = np.zeros((num_classes,),dtype=float)
      zer[curr_answer] = 1 

    img = cv2.imread(os.path.join(self.dataset_dir, 'Images/', curr_filename + '.png'))

    
    # Resize image and mask
    img = cv2.resize(img, (int(700/2),int(400/2)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img.resize((700,400))
    img_arr = np.array(img)


    return (curr_question,  img_arr), zer