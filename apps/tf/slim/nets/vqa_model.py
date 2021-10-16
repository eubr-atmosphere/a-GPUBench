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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

"""
Implementation of the model for Visual Question Answering.
The original paper can be found at: https://arxiv.org/abs/1505.00468

For the CNN part is used an Xception instead of a VGG, as described in the paper.
"""



def vqa(num_classes=58,
          is_training=True,
          dropout_keep_prob=0.3,
          network_depth = 0,
          weight_decay = None):

    # Creating Model
    # --------------
    EMBEDDING_SIZE = 128

    # Constants from datasets
    max_questions_length = 21
    questions_wtoi =  4640

    if weight_decay == None:
     kernel_regularizer = None
    else:
     kernel_regularizer = tf.keras.regularizers.L2(weight_decay)   


    # LSTM: parameters are set to use cuDNN implementation and speed up training

    encoder_input = tf.keras.Input(shape=[max_questions_length])
    encoder_embedding_layer = tf.keras.layers.Embedding(input_dim = questions_wtoi+1, output_dim = 512, input_length=max_questions_length, mask_zero=True)(encoder_input)
    encoded_question = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBEDDING_SIZE, unroll=False, recurrent_dropout=0, activation="tanh", recurrent_activation="sigmoid", kernel_regularizer = kernel_regularizer, use_bias=True))(encoder_embedding_layer)



    # CNN 

    model_img = tf.keras.models.Sequential()
    xception = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=[200,350, 3])

    finetuning = True

    if finetuning:
     freeze_until = 16 # layer from which we want to fine-tune

    for layer in xception.layers:
     layer.trainable = False

    for layer in xception.layers[:freeze_until]:
     layer.trainable = True

    model_img.add(xception)
    model_img.add(tf.keras.layers.Flatten())
    model_img.add(tf.keras.layers.Dense(EMBEDDING_SIZE, activation="relu",  kernel_regularizer = kernel_regularizer))
    model_img.add(tf.keras.layers.Dropout(dropout_keep_prob))

    image_input = tf.keras.Input(shape=[200,350, 3])
    out_image = model_img(image_input)


    vqa_model_mul = tf.keras.layers.concatenate(inputs = [out_image, encoded_question])
    vqa_model_fc = tf.keras.layers.Dense(128, activation="relu",  kernel_regularizer = kernel_regularizer)(vqa_model_mul)
    vqa_model_out = tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer = kernel_regularizer)(vqa_model_fc)


    model = tf.keras.Model(inputs =  [encoder_input, image_input], outputs = vqa_model_out)

    return model
