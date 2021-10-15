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
"""Generic training script that trains a model using a given dataset."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa


from tensorflow.quantization import quantize as contrib_quantize

import argparse
import os

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import logging
import time
import datetime

parser = argparse.ArgumentParser()

 
parser.add_argument(
    '--master', default = '', help = 'The address of the TensorFlow master to use.')

parser.add_argument(
    '--warmup_epochs', default = 0, type=int,
    help = 'Linearly warmup learning rate from 0 to learning_rate over this '
    'many epochs.')
parser.add_argument(
  '--number_of_epochs', default=1, type=int,
  help= 'Number of epochs'
)    

parser.add_argument(
  '--multiple_gpu', default = False, type = bool, help = 'run training on mulitple gpu'
)

parser.add_argument(
  '--patience', default = None, type = int, help = "Patience before early stopping"
)

parser.add_argument(
  '--network_depth', default = None, type = int, help = "Depth of the network; Used with variable depth netoworks"
)

parser.add_argument(
    '--train_dir', default = '/tmp/tfmodel/',
    help = 'Directory where checkpoints and event logs are written to.')

parser.add_argument(
    '--logdir', default = '/tmp/tflog/',
    help = 'Directory where traces are written to.')

parser.add_argument(
    '--trace', default=   False, type = bool,
    help = 'Generate traces.')

parser.add_argument('--num_clones',default =  1, type = int,
                            help = 'Number of model clones to deploy.')

parser.add_argument('--clone_on_cpu', default = False, type = bool,
                            help = 'Use CPUs to deploy clones.')

parser.add_argument('--worker_replicas', default =  1, type = int, help =  'Number of worker replicas.')

parser.add_argument(
    '--num_ps_tasks', default = 0, type = int,
    help = 'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

parser.add_argument(
    '--num_readers', default = 4, type = int, 
    help = 'The number of parallel readers that read data from the dataset.')

parser.add_argument(
    '--num_preprocessing_threads', default = 4, type = int,
    help = 'The number of threads used to create the batches.')

parser.add_argument(
    '--log_every_n_steps', default = 10, type = int,
    help = 'The frequency with which logs are print.')

parser.add_argument(
    '--save_summaries_secs', default = 600, type = int,
    help = 'The frequency with which summaries are saved, in seconds.')

parser.add_argument(
    '--save_interval_secs',  default = 600, type = int,
    help = 'The frequency with which the model is saved, in seconds.')

parser.add_argument(
    '--task', default = 0, type = int, help= 'Task id of the replica running the training.')

parser.add_argument(
  '--perform_validation', default = False, type = bool, help='Perform evalutation on validation set during training'
)

parser.add_argument(
  '--model_out', default = False, help='Directory where to put saved model at the end of the training session'
)

parser.add_argument(
  '--seed', default=0, type = int, help =  'seed'
)
######################
# Optimization Flags #
######################

parser.add_argument(
    '--weight_decay', default = 0.00004, type=float,  help='The weight decay on the model weights.')
parser.add_argument(
  '--dropout', default = 0.2, type = float, help = 'Dropout value'
)    

parser.add_argument(
    '--optimizer', default = 'AdamW',
    help = 'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

parser.add_argument(
    '--adadelta_rho', default = 0.95, type = float,
    help = 'The decay rate for adadelta.')

parser.add_argument(
    '--adagrad_initial_accumulator_value',default =  0.1, type = float,
    help = 'Starting value for the AdaGrad accumulators.')

parser.add_argument(
    '--adam_beta1', default = 0.9, type = float,
    help = 'The exponential decay rate for the 1st moment estimates.')

parser.add_argument(
    '--adam_beta2', default = 0.999, type = float,
    help = 'The exponential decay rate for the 2nd moment estimates.')

parser.add_argument('--opt_epsilon', default = 1.0, type = float, help = 'Epsilon term for the optimizer.')

parser.add_argument('--ftrl_learning_rate_power', default = -0.5, type = float,
                         help = 'The learning rate power.')

parser.add_argument(
    '--ftrl_initial_accumulator_value', default =  0.1, type=float,
    help = 'Starting value for the FTRL accumulators.')

parser.add_argument(
    '--ftrl_l1',default= 0.0, type = float, help =  'The FTRL l1 regularization strength.')

parser.add_argument(
    '--ftrl_l2', default= 0.0, type=float , help ='The FTRL l2 regularization strength.')

parser.add_argument(
    '--momentum',default =  0.9, type = float,
    help = 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

parser.add_argument('--rmsprop_momentum', default = 0.9, type=float, help = 'Momentum.')

parser.add_argument('--rmsprop_decay', default = 0.9, type=float,help= 'Decay term for RMSProp.')

parser.add_argument(
    '--quantize_delay', default = -1, type = int,
    help = 'Number of steps to start quantized training. Set to -1 would disable '
    'quantized training.')
parser.add_argument(
    '--clipvalue', default = 10, type = int,
    help = 'clipping value used in order to avoid exploding gradient')    

#######################
# Learning Rate Flags #
#######################

parser.add_argument(
    '--learning_rate_decay_type',
    default='exponential',
    help = 'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

parser.add_argument('--learning_rate', default = 0.01, type = float, help= 'Initial learning rate.')

parser.add_argument(
    '--end_learning_rate', default= 0.0001, type=float,
    help = 'The minimal end learning rate used by a polynomial decay learning rate.')

parser.add_argument(
    '--label_smoothing', default = 0.0, type=float, help = 'The amount of label smoothing.')

parser.add_argument(
    '--learning_rate_decay_factor',default= 0.94, type = float, help =  'Learning rate decay factor.')

parser.add_argument(
    '--num_epochs_per_decay', default = 2.0, type=float,
    help = 'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

parser.add_argument(
    '--sync_replicas', default =  False, type = bool,
    help = 'Whether or not to synchronize the replicas during training.')

parser.add_argument(
    '--replicas_to_aggregate', default = 1, type = int,
    help = 'The Number of gradients to collect before updating params.')

parser.add_argument(
    '--moving_average_decay', default =  None, type=float,
    help = 'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

parser.add_argument(
    '--dataset_name', default = 'flowers', help = 'The name of the dataset to load.')

parser.add_argument(
  '--num_classes', default = 5, type=int, help=" Number of classes in the dataset")    

parser.add_argument(
    '--dataset_split_name', default = 'train', help =  'The name of the train/test split.')

parser.add_argument(
    '--dataset_dir', default = "/tmp/flowers_2/5", help =  'The directory where the dataset files are stored.')

parser.add_argument(
    '--labels_offset', default = 0, type = int,
    help = 'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

parser.add_argument(
    '--model_name', default = 'alexnet_v2', help =  'The name of the architecture to train.')

parser.add_argument(
    '--preprocessing_name', default =  None, help = 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

parser.add_argument(
    '--batch_size', default= 32, type = int, help =  'The number of samples in each batch.')

parser.add_argument(
    '--train_image_size', default = None, type = int, help =  'Train image size')

parser.add_argument('--max_number_of_steps', default = None, type = int,
                            help='The maximum number of training steps.')

parser.add_argument('--use_grayscale', default = False, type = bool,
                         help = 'Whether to convert input images to grayscale.')

#####################
# Fine-Tuning Flags #
#####################

parser.add_argument(
    '--checkpoint_path', default= None,
    help = 'The path to a checkpoint from which to fine-tune.')

parser.add_argument(
    '--checkpoint_exclude_scopes', default = None,
    help ='Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

parser.add_argument(
    '--trainable_scopes',default =  None,
    help = 'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

parser.add_argument(
    '--ignore_missing_vars', default = False, type = bool,
    help = 'When restoring a checkpoint would ignore missing variables.')

args = parser.parse_args()

#
# Configuration of the learning rate
# This code can be reused and adapted 
#
def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  # Note: when num_clones is > 1, this will actually have each clone to go
  # over each epoch num_epochs_per_decay times. This is different
  # behavior from sync replicas and is expected to produce different results.
  args.steps_per_epoch = num_samples_per_epoch / args.batch_size
  if args.sync_replicas:
    args.steps_per_epoch /= args.replicas_to_aggregate

  args.decay_steps = int(args.steps_per_epoch * args.num_epochs_per_decay)

  if args.learning_rate_decay_type == 'exponential':
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate,
        args.decay_steps,
        args.learning_rate_decay_factor,
        staircase=True,
        name='exponential_decay_learning_rate')
  elif args.learning_rate_decay_type == 'fixed':
    learning_rate = tf.constant(args.learning_rate, name='fixed_learning_rate')
  elif args.learning_rate_decay_type == 'polynomial':
    learning_rate = tf.optimizers.schedules.PolynomialDecay(
        args.learning_rate,
        args.decay_steps,
        args.end_learning_rate,
        power=1.0,
        cycle=False,
        name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                     args.learning_rate_decay_type)

  if args.warmup_epochs:
    warmup_lr = (
        learning_rate * tf.cast(global_step, tf.float32) /
        (args.steps_per_epoch * args.warmup_epochs))
    learning_rate = tf.minimum(warmup_lr, learning_rate)
  return learning_rate


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if optimizer is not recognized.
  """
  if args.optimizer == 'adadelta':
    optimizer = tf.keras.optimizers.Adadelta(
        learning_rate,
        rho=args.adadelta_rho,
        epsilon=args.opt_epsilon,
        clipvalue= args.clipvalue)
  elif args.optimizer == 'adagrad':
    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate,
        initial_accumulator_value=args.adagrad_initial_accumulator_value,
        clipvalue= args.clipvalue)
  elif args.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=args.adam_beta1,
        beta_2=args.adam_beta2,
        epsilon=args.opt_epsilon,
        clipvalue= args.clipvalue)
  elif args.optimizer == "AdamW":
    optimizer = tfa.optimizers.AdamW(args.weight_decay,
        learning_rate=learning_rate,
        beta_1 = args.adam_beta1,
        beta_2 = args.adam_beta2,
        epsilon = args.opt_epsilon,
        clipvalue= args.clipvalue)      
  elif args.optimizer == 'ftrl':
    optimizer = tf.keras.optimizers.Ftrl(
        learning_rate,
        learning_rate_power=args.ftrl_learning_rate_power,
        initial_accumulator_value=args.ftrl_initial_accumulator_value,
        l1_regularization_strength=args.ftrl_l1,
        l2_regularization_strength=args.ftrl_l2,
        clipvalue= args.clipvalue)
  elif args.optimizer == 'momentum':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate,
        momentum=args.momentum,
        name='Momentum',
        clipvalue= args.clipvalue)
  elif args.optimizer == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate,
        decay=args.rmsprop_decay,
        momentum=args.rmsprop_momentum,
        epsilon=args.opt_epsilon,
        clipvalue= args.clipvalue)
  elif args.optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate, clipvalue= args.clipvalue)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % optimizer)  
  return optimizer


def main():
  if not args.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not args.model_out:
    raise ValueError('You must supply where put saved model with --model_out') 
  if args.patience != None and args.perform_validation == False:
    raise ValueError("Patience is setted, but validation is set to False")  

  if(args.network_depth == -1):
    args.network_depth = None   

  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  global_step = tf.Variable(1, name="global_step")

  ######################
  # Select the dataset #
  ######################
  num_classes, num_samples, num_validation_samples, training_set, validation_set = dataset_factory.get_dataset(
      args.dataset_name, args.dataset_split_name, args.dataset_dir, args.seed, args.batch_size)
  

  
  ######################
  # Select the network #
  ######################
  network_fn = nets_factory.get_network_fn(
      args.model_name,
      num_classes=(num_classes - args.labels_offset),
      weight_decay=args.weight_decay,
      is_training=True,
      network_depth = args.network_depth)

  #####################################
  # Select the preprocessing function #
  #####################################
  preprocessing_name = args.preprocessing_name or args.model_name
  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      preprocessing_name,
      is_training=True)

  if image_preprocessing_fn != None:
   logging.info("Apply preprocessing function") 
   training_set = training_set.map(lambda x, y: (image_preprocessing_fn(x), y))

  #########################################
  # Configure the optimization procedure. #
  #########################################
  
  learning_rate = _configure_learning_rate(num_samples, global_step)

  opt = _configure_optimizer(learning_rate)
  
  loss = tf.keras.losses.CategoricalCrossentropy() 


  ###########################
  # Kicks off the training. #
  ###########################
  
  #callbacks to pass to fit
  callbacks = []
  
  #Append TimeCallback used to print iterations
  callbacks.append(TimeCallback())
  
  network_fn.compile(optimizer=opt, loss = loss, metrics=["accuracy"])
  
  #If patience is set, add the early stopping callback to fit 
  #Default restoring best weights
  if args.patience != None:
    if args.dataset_name != "vqa":
     callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights = True))
    else:
     #Setting min delta to 0.01 in order to speedup the training of vqa model 
     callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.01, patience=args.patience, restore_best_weights = True)) 
  #network_fn.summary()

  if args.perform_validation == True:
      logging.info("Performing validation")
      history = network_fn.fit(x = training_set, validation_data = validation_set, validation_steps = int(num_validation_samples/args.batch_size) , epochs=args.number_of_epochs, verbose=0, callbacks=callbacks, steps_per_epoch=int(num_samples/args.batch_size), batch_size = args.batch_size)
  else: 
      logging.info("Start training")
      history = network_fn.fit(x = training_set, epochs=args.number_of_epochs, verbose=0, callbacks=callbacks, steps_per_epoch=int(num_samples/args.batch_size), batch_size = args.batch_size)
  #print(min(int(num_samples//args.batch_size), args.max_number_of_steps))
  
  #save weights of the model on the disk
  logging.info("Saving weight of the model on " + args.model_out) 
  network_fn.save_weights(os.path.join(args.model_out, args.model_name+"_"+args.dataset_name+"_"+str(args.batch_size)+"_"+str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))+"/"+"weights_ckp"))
  logging.info("Weights saved") 
  
# Define a custom callback to pass into fit in order to print loss and step time
class TimeCallback(tf.keras.callbacks.Callback):
 def on_train_batch_end(self, batch, logs=None):
  #keys = list(logs.keys())
  curr_step = (self.epoch_number * self.params['steps']) + batch+1
  logging.info("tensorflow:global step {0}: loss = {1:.4f} ({2:.3f} sec/step)".format(curr_step, logs['loss'], time.time() - self.start))


 # save timestamp of when start training on batch in order to compute the time needed
 def on_train_batch_begin(self, batch, logs=None):
  self.start = time.time()
  
 #save epoch number in order to determin global INFO: : p 
 def on_epoch_begin(self, epoch, logs=None):
   self.epoch_number = epoch

 #Logs loss and accuracy of the model on the validation set
 def on_test_end(self, logs=None): 
  logging.info("tensorflow: evaluation on validation set: loss = {0:.4f}, accuracy = {1:.4f}".format(logs['loss'], logs['accuracy']))





if __name__ == '__main__':
  main()
