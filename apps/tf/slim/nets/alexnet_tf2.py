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
"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



def alexnet_v2(num_classes=1000,
               is_training=True,
               weight_decay = 0.0005,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2',
               network_depth = None,
               global_pool=False):
  """AlexNet version 2.

  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224 or set
        global_pool=True. To use in fully convolutional mode, set
        spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: the number of predicted classes. If 0 or None, the logits layer
    is omitted and the input features to the logits layerdropout_keep_prob are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      logits. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original AlexNet.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0
      or None).
    end_points: a dict of tensors with intermediate activations.
  """
  if weight_decay == None:
   regularizer = None
  else:
   regularizer = tf.keras.regularizers.L2(weight_decay) 

  model = tf.keras.Sequential()
  
  model.add(tf.keras.Input(shape=[224, 224, 3]))
  model.add(tf.keras.layers.Conv2D(96, (11,11), (4,4), kernel_regularizer=regularizer))
  model.add(tf.keras.layers.ReLU())
  model.add(tf.keras.layers.MaxPool2D((2,2), (2,2)))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(256, (11, 11), (1,1), kernel_regularizer=regularizer))
  model.add(tf.keras.layers.ReLU())
  model.add(tf.keras.layers.MaxPool2D((3,3), (2,2)))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(384, (3,3), (1,1), kernel_regularizer=regularizer))
  model.add(tf.keras.layers.ReLU())
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(384, (3,3), (1,1), kernel_regularizer=regularizer))
  model.add(tf.keras.layers.ReLU())
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(256, (3,3), (1,1), kernel_regularizer=regularizer))
  model.add(tf.keras.layers.ReLU())
  model.add(tf.keras.layers.MaxPool2D((2,2), (2,2)))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Flatten())

  #Fully Connected Part
  model.add(tf.keras.layers.Dense(4096, kernel_regularizer=regularizer))
  model.add(tf.keras.layers.ReLU())
    
  model.add(tf.keras.layers.Dropout(dropout_keep_prob))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Dense(4096, activation="relu", kernel_regularizer=regularizer))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dense(1000, kernel_regularizer=regularizer))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(dropout_keep_prob))
  model.add(tf.keras.layers.BatchNormalization())
  # Output Layer
  model.add(tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizer))

  return model   

alexnet_v2.default_image_size = 224
