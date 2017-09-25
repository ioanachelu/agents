# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Networks for the PPO algorithm defined as recurrent cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

_MEAN_WEIGHTS_INITIALIZER = tf.contrib.layers.variance_scaling_initializer(
    factor=0.1)
_LOGSTD_INITIALIZER = tf.random_normal_initializer(-1, 1e-10)


class ForwardGaussianPolicy(tf.contrib.rnn.RNNCell):
  """Independent feed forward networks for policy and value.

  The policy network outputs the mean action and the log standard deviation
  is learned as independent parameter vector.
  """

  def __init__(
      self, policy_layers, value_layers, action_size,
      mean_weights_initializer=_MEAN_WEIGHTS_INITIALIZER,
      logstd_initializer=_LOGSTD_INITIALIZER):
    self._policy_layers = policy_layers
    self._value_layers = value_layers
    self._action_size = action_size
    self._mean_weights_initializer = mean_weights_initializer
    self._logstd_initializer = logstd_initializer

  @property
  def state_size(self):
    unused_state_size = 1
    return unused_state_size

  @property
  def output_size(self):
    return (self._action_size, self._action_size, tf.TensorShape([]))

  def __call__(self, observation, state):
    with tf.variable_scope('policy'):
      x = tf.contrib.layers.flatten(observation)
      for size in self._policy_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      mean = tf.contrib.layers.fully_connected(
          x, self._action_size, tf.tanh,
          weights_initializer=self._mean_weights_initializer)
      logstd = tf.get_variable(
          'logstd', mean.shape[1:], tf.float32, self._logstd_initializer)
      logstd = tf.tile(
          logstd[None, ...], [tf.shape(mean)[0]] + [1] * logstd.shape.ndims)
    with tf.variable_scope('value'):
      x = tf.contrib.layers.flatten(observation)
      for size in self._value_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      value = tf.contrib.layers.fully_connected(x, 1, None)[:, 0]
    return (mean, logstd, value), state

class RecurrentGaussianPolicy(tf.contrib.rnn.RNNCell):
  """Independent recurrent policy and feed forward value networks.

  The policy network outputs the mean action and the log standard deviation
  is learned as independent parameter vector. The last policy layer is recurrent
  and uses a GRU cell.
  """

  def __init__(
      self, policy_layers, value_layers, action_size,
      mean_weights_initializer=_MEAN_WEIGHTS_INITIALIZER,
      logstd_initializer=_LOGSTD_INITIALIZER):
    self._policy_layers = policy_layers
    self._value_layers = value_layers
    self._action_size = action_size
    self._mean_weights_initializer = mean_weights_initializer
    self._logstd_initializer = logstd_initializer
    self._cell = tf.contrib.rnn.GRUBlockCell(100)

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return (self._action_size, self._action_size, tf.TensorShape([]))

  def __call__(self, observation, state):
    with tf.variable_scope('policy'):
      x = tf.contrib.layers.flatten(observation)
      for size in self._policy_layers[:-1]:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      x, state = self._cell(x, state)
      mean = tf.contrib.layers.fully_connected(
          x, self._action_size, tf.tanh,
          weights_initializer=self._mean_weights_initializer)
      logstd = tf.get_variable(
          'logstd', mean.shape[1:], tf.float32, self._logstd_initializer)
      logstd = tf.tile(
          logstd[None, ...], [tf.shape(mean)[0]] + [1] * logstd.shape.ndims)
    with tf.variable_scope('value'):
      x = tf.contrib.layers.flatten(observation)
      for size in self._value_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      value = tf.contrib.layers.fully_connected(x, 1, None)[:, 0]
    return (mean, logstd, value), state

class AOCPolicy(tf.contrib.rnn.RNNCell):

  def __init__(
      self, conv_layers, fc_layers, action_size, nb_options):
    self._conv_layers = conv_layers
    self._fc_layers = fc_layers
    self._action_size = action_size
    self._nb_options = nb_options

  @property
  def state_size(self):
    unused_state_size = 1
    return unused_state_size

  @property
  def output_size(self):
    return (1, 1, tf.TensorShape([]))

  def __call__(self, observation, state):
    with tf.variable_scope('conv'):
      for kernel_size, stride, nb_kernels in self._conv_layers:
        out = layers.conv2d(observation, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=tf.nn.relu,
                            variables_collections=tf.get_collection("variables"),
                            outputs_collections="activations")
      out = layers.flatten(out)
      with tf.variable_scope("fc"):
        for nb_filt in self._fc_layers:
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                           activation_fn=None,
                                           variables_collections=tf.get_collection("variables"),
                                           outputs_collections="activations")
          out = layer_norm_fn(out, relu=True)

      self.termination = layers.fully_connected(out, num_outputs=self._nb_options,
                                                                activation_fn=tf.nn.sigmoid,
                                                                variables_collections=tf.get_collection("variables"),
                                                                outputs_collections="activations")
      self.q_val = layers.fully_connected(out, num_outputs=self._nb_options,
                                                  activation_fn=None,
                                                  variables_collections=tf.get_collection("variables"),
                                                  outputs_collections="activations")
      self.options = []
      for _ in range(self._nb_options):
        option = layers.fully_connected(out, num_outputs=self._action_size,
                                            activation_fn=tf.nn.softmax,
                                            variables_collections=tf.get_collection("variables"),
                                            outputs_collections="activations")
        self.options.append(option)

    return (self.termination, self.q_val, self.options), state

def layer_norm_fn(x, relu=True):
  x = layers.layer_norm(x, scale=True, center=True)
  if relu:
    x = tf.nn.relu(x)
  return x