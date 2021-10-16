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
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def vgg_16(num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           weight_decay = 0.0005,
           reuse=None,
           network_depth = None,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  if weight_decay == None:
     kernel_regularizer = None
  else:
     kernel_regularizer = tf.keras.regularizers.L2(weight_decay) 

  model = tf.keras.Sequential()
  filter_n = 64

  model.add(tf.keras.Input(shape=[224, 224, 3]))
  #Feature extraction part
  for i in range(4):
   model.add(tf.keras.layers.Conv2D(filter_n, (3,3), padding='same', kernel_regularizer=kernel_regularizer))
   model.add(tf.keras.layers.ReLU())
   model.add(tf.keras.layers.Conv2D(filter_n, (3,3), padding='same',kernel_regularizer=kernel_regularizer))
   model.add(tf.keras.layers.ReLU())
   if i > 1:
     model.add(tf.keras.layers.Conv2D(filter_n, (3,3),padding='same', kernel_regularizer=kernel_regularizer))
     model.add(tf.keras.layers.ReLU())

   model.add(tf.keras.layers.MaxPool2D((2,2), strides = 2))
   filter_n *= 2  
  
  model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same', kernel_regularizer=kernel_regularizer))
  model.add(tf.keras.layers.ReLU())
  model.add(tf.keras.layers.Conv2D(512, (3,3), padding='same',kernel_regularizer=kernel_regularizer))
  model.add(tf.keras.layers.ReLU())
  model.add(tf.keras.layers.Conv2D(512, (3,3),padding='same', kernel_regularizer=kernel_regularizer))
  model.add(tf.keras.layers.ReLU())

  model.add(tf.keras.layers.MaxPool2D((2,2), strides = 2))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(4096, activation="relu", kernel_regularizer = kernel_regularizer))
  model.add(tf.keras.layers.Dropout(dropout_keep_prob))
  model.add(tf.keras.layers.Dense(4096, activation="relu", kernel_regularizer = kernel_regularizer))
  model.add(tf.keras.layers.Dropout(dropout_keep_prob))
  model.add(tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer = kernel_regularizer))

  return model   
vgg_16.default_image_size = 224


def vgg_19(num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           weight_decay = None,
           spatial_squeeze=True,
           network_depth = None,
           global_pool=False):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations.
  """
  if weight_decay == None:
     kernel_regularizer = None
  else:
     kernel_regularizer = tf.keras.regularizers.L2(weight_decay) 

  model = tf.keras.Sequential()
  filter_n = 64

  model.add(tf.keras.Input(shape=[224, 224, 3]))
  #Feature extraction part
  for i in range(5):
   model.add(tf.keras.layers.Conv2D(filter_n, (3,3), kernel_regularizer=kernel_regularizer))
   model.add(tf.keras.layers.ReLU())
   model.add(tf.keras.layers.Conv2D(filter_n, (3,3), kernel_regularizer=kernel_regularizer))
   model.add(tf.keras.layers.ReLU())
   if i > 1:
     model.add(tf.keras.layers.Conv2D(filter_n, (3,3), kernel_regularizer=kernel_regularizer))
     model.add(tf.keras.layers.ReLU())
     model.add(tf.keras.layers.Conv2D(filter_n, (3,3), kernel_regularizer=kernel_regularizer))
     model.add(tf.keras.layers.ReLU())

   model.add(tf.keras.layers.MaxPool2D((2,2), strides = 2))
   filter_n *= 2  
  
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(4096, activation="relu", kernel_regularizer = kernel_regularizer))
  model.add(tf.keras.layers.Dropout(dropout_keep_prob))
  model.add(tf.keras.layers.Dense(4096, activation="relu", kernel_regularizer = kernel_regularizer))
  model.add(tf.keras.layers.Dropout(dropout_keep_prob))
  model.add(tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer = kernel_regularizer))

  return model   
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19