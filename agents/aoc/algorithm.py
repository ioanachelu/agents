
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections

from agents.aoc import memory
from agents.aoc import normalize
from agents.aoc import utility


_NetworkOutput = collections.namedtuple(
    'NetworkOutput', 'termination, q_val, options, state')

class AOCAlgorithm(object):

  def __init__(self, batch_env, step, is_training, should_log, config):
    self._batch_env = batch_env
    self._step = step
    self._is_training = is_training
    self._should_log = should_log
    self._config = config
    self._observ_filter = normalize.StreamingNormalize(
      self._batch_env.observ[0], center=True, scale=True, clip=5,
      name='normalize_observ')
    self._reward_filter = normalize.StreamingNormalize(
      self._batch_env.reward[0], center=False, scale=True, clip=10,
      name='normalize_reward')
    # Memory stores tuple of observ, action, mean, logstd, reward.
    template = (
      self._batch_env.observ[0], self._batch_env.action[0],
      self._batch_env.reward[0], self._batch_env.observ[0],
      self._batch_env.reward[0])
    self._memory = memory.EpisodeMemory(
      template, config.update_every, config.max_length, 'memory')
    self._memory_index = tf.Variable(0, False)
    use_gpu = self._config.use_gpu and utility.available_gpus()
    with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
      self._network(
        tf.zeros_like(self._batch_env.observ)[:, None],
        tf.ones(len(self._batch_env)), reuse=None)
      cell = self._config.network(self._batch_env._batch_env._envs[0]._action_space.n)

      self._network_optimizer = self._config.network_optimizer(
        self._config.lr, name='network_optimizer')

  def begin_episode(self, unused_agent_indices):
    return tf.constant('')

  def perform(self, unused_observ):
    shape = (len(self._envs),) + self._envs[0].action_space.shape
    low = self._envs[0].action_space.low
    high = self._envs[0].action_space.high
    action = tf.random_uniform(shape) * (high - low) + low
    return action, tf.constant('')

  def experience(self, *unused_transition):
    return tf.constant('')

  def end_episode(self, unused_agent_indices):
    return tf.constant('')

  def _network(self, observ, length=None, state=None, reuse=True):
    with tf.variable_scope('network', reuse=reuse):
      observ = tf.convert_to_tensor(observ)
      use_gpu = self._config.use_gpu and utility.available_gpus()
      with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
        observ = tf.check_numerics(observ, 'observ')
        cell = self._config.network(self._batch_env._batch_env._envs[0]._action_space.n)
        (termination, q_val, options), state = tf.nn.dynamic_rnn(
            cell, observ, length, state, tf.float32, swap_memory=True)

      return _NetworkOutput(termination, q_val, options, state)

