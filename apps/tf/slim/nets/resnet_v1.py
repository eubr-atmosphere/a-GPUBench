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
"""Contains definitions for the original form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tf_slim.nets import resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
credits: https://medium.com/analytics-vidhya/understanding-and-implementation-of-residual-networks-resnets-b80f9a507b9c
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def resnet_v1_50(num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           weight_decay = 0.0005,
           reuse=None,
           fc_conv_padding='VALID',
           network_depth = None,
           global_pool=False):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode. If this is set
      to None, the callers can specify slim.batch_norm's is_training parameter
      from an outer slim.arg_scope.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        To use this parameter, the input images must be smaller than 300x300
        pixels, in which case the output logit layer does not contain spatial
        information and can be removed.
    store_non_strided_activations: If True, we compute non-strided (undecimated)
      activations at the last unit of each block and store them in the
      `outputs_collections` before subsampling them. This gives us access to
      higher resolution intermediate activations which are useful in some
      dense prediction problems but increases 4x the computation and memory cost
      at the last unit of each block.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes a non-zero integer, net contains the
      pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
  X_input = tf.keras.Input(shape=[224, 224, 3])

  if weight_decay == None:
   kernel_regularizer = None
  else:
   kernel_regularizer = tf.keras.regularizers.L2(weight_decay) 
  
  if network_depth != None:
   initial_size = network_depth
  else: 
   initial_size = 3 #Defalut size is 3, as the original resnet 50

  starting_size = initial_size  
  # Zero-Padding
  X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
    
  # Stage 1
  X = tf.keras.layers.Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_regularizer=kernel_regularizer)(X)
  X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
  X = tf.keras.layers.Activation('relu')(X)
  X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

  # Stage 2
  X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='0', s = 1, kernel_regularizer = kernel_regularizer)
  for i in range(initial_size-1):
   X = identity_block(X, 3, [64, 64, 256], stage=2, block="a"+str(i), kernel_regularizer = kernel_regularizer)
  #X = identity_block(X, 3, [64, 64, 256], stage=2, block='c',kernel_regularizer = kernel_regularizer)

  initial_size = initial_size + 1
  # Stage 3 
  X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='1', s = 2, kernel_regularizer = kernel_regularizer)
  for i in range(initial_size-1):
   X = identity_block(X, 3, [128, 128, 512], stage=3, block="b"+str(i), kernel_regularizer = kernel_regularizer)
  #X = identity_block(X, 3, [128, 128, 512], stage=3, block='c', kernel_regularizer = kernel_regularizer)
  #X = identity_block(X, 3, [128, 128, 512], stage=3, block='d', kernel_regularizer = kernel_regularizer)
   
  initial_size = (starting_size) * 2 
  # Stage 4 
  X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2, kernel_regularizer = kernel_regularizer)
  for i in range(initial_size - 1):
   X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c'+str(i),kernel_regularizer = kernel_regularizer)
  #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c',kernel_regularizer = kernel_regularizer)
  #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d',kernel_regularizer = kernel_regularizer)
  #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e',kernel_regularizer = kernel_regularizer)
  #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f',kernel_regularizer = kernel_regularizer)

  # Stage 5 
  X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='2', s = 2, kernel_regularizer = kernel_regularizer)
  for i in range(starting_size - 1):
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='d'+str(i), kernel_regularizer = kernel_regularizer)
  #X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c', kernel_regularizer = kernel_regularizer)

  # AVGPOOL . Use "X = AveragePooling2D(...)(X)"
  X = tf.keras.layers.AveragePooling2D()(X)

  # output layer
  X = tf.keras.layers.Flatten()(X)
  X = tf.keras.layers.Dense(num_classes, activation='softmax', name='fc' + str(num_classes), kernel_regularizer = kernel_regularizer)(X)
    
    
  # Create model
  model = tf.keras.Model(inputs = X_input, outputs = X, name='ResNet50')

  return model

def convolutional_block(X, f, filters, stage, block, s = 2, kernel_regularizer = None):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = tf.keras.layers.Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_regularizer = kernel_regularizer)(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)
    

    # Second component of main path 
    X = tf.keras.layers.Conv2D(F2, (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_regularizer = kernel_regularizer)(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu') (X)

    # Third component of main path 
    X = tf.keras.layers.Conv2D(F3, (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_regularizer = kernel_regularizer)(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = tf.keras.layers.Conv2D(F3, (1,1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_regularizer = kernel_regularizer)(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    
    return X

def identity_block(X, f, filters, stage, block, kernel_regularizer = None):
  """
  Implementation of the identity block
  
  Arguments:
  X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
  f -- integer, specifying the shape of the middle CONV's window for the main path
  filters -- python list of integers, defining the number of filters in the CONV layers of the main path
  stage -- integer, used to name the layers, depending on their position in the network
  block -- string/character, used to name the layers, depending on their position in the network
  
  Returns:
  X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
  """
  
  # defining name basis
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  
  # Retrieve Filters
  F1, F2, F3 = filters
  
  # Save the input value. You'll need this later to add back to the main path. 
  X_shortcut = X
  
  # First component of main path
  X = tf.keras.layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_regularizer = kernel_regularizer)(X)
  X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
  X = tf.keras.layers.Activation('relu')(X)
  
  
  # Second component of main path
  X = tf.keras.layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_regularizer = kernel_regularizer)(X)
  X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
  X = tf.keras.layers.Activation('relu')(X)

  # Third component of main path 
  X = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_regularizer = kernel_regularizer)(X)
  X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

  # Final step: Add shortcut value to main path, and pass it through a RELU activation 
  X = tf.keras.layers.Add()([X, X_shortcut])
  X = tf.keras.layers.Activation('relu')(X)
  
  
  return X  

resnet_v1_50.default_image_size = 224
