
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections

from agents.aoc import memory
from agents.aoc import normalize
from agents.aoc import utility
import numpy as np

from agents.aoc.schedules import LinearSchedule


_NetworkOutput = collections.namedtuple(
    'NetworkOutput', 'termination, q_val, options, state')

class AOCAlgorithm(object):

  def __init__(self, batch_env, step, is_training, should_log, config):
    self._batch_env = batch_env
    self._step = step
    self._is_training = is_training
    self._should_log = should_log

    self._config = config
    self._frame_counter = 0
    self._nb_options = config.nb_options
    self._action_size = self._batch_env._batch_env._envs[0]._action_space.n
    self._num_agents = self._config.num_agents
    self._option_terminated = np.asarray(self._num_agents * [True])


    self._random = tf.random_uniform(shape=[(self._config.num_agents)], minval=0., maxval=1., dtype=tf.float32)
    self._exploration_options = LinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                               self._config.initial_random_action_prob)
    self._exploration_policies = LinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                                self._config.initial_random_action_prob)
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
    with tf.variable_scope('aoc_temporary'):
      self._episodes = memory.EpisodeMemory(
        template, len(batch_env), config.max_length, 'episodes')
      self._last_state = utility.create_nested_vars(
        cell.zero_state(len(batch_env), tf.float32))
      self._last_action = tf.Variable(
        tf.zeros_like(self._batch_env.action), False, name='last_action')

  def begin_episode(self, agent_indices):
    with tf.name_scope('begin_episode/'):
      reset_state = utility.reinit_nested_vars(self._last_state, agent_indices)
      reset_buffer = self._episodes.clear(agent_indices)
      self._delib_cost = self._num_agents * [self._config.delib_cost]

      with tf.control_dependencies([reset_state, reset_buffer]):
        return tf.constant('')

  def perform(self, observ):
    with tf.name_scope('perform/'):
      observ = self._observ_filter.transform(observ)
      network = self._network(
        observ[:, None], tf.ones(observ.shape[0]), self._last_state)

      self._current_option[self._option_terminated] = self.get_policy_over_options(network)[self._option_terminated]

      action = self.get_action(network)

    return tf.cast(action, dtype=tf.int32) , tf.constant('')

  def experience(self, observ, action, reward, done, nextob):
    with tf.name_scope('experience/'):
      return tf.cond(
        self._is_training,
        lambda: self._define_experience(observ, action, reward, done, nextob), str)

  def _define_experience(self, observ, action, reward, done, nextob):
    """Implement the branch of experience() entered during training."""
    update_filters = tf.summary.merge([
      self._observ_filter.update(observ),
      self._reward_filter.update(reward)])

    with tf.control_dependencies([update_filters]):
      # if self._config.train_on_agent_action:
      #   # NOTE: Doesn't seem to change much.
      #   action = self._last_action
      self._frame_counter += 1
      self.new_reward = reward - \
                        np.asarray(self._option_terminated, dtype=np.float32) * \
                        self._delib_cost * float()


      append = self._episodes.append(batch, tf.range(len(self._batch_env)))
    with tf.control_dependencies([append]):
      norm_observ = self._observ_filter.transform(observ)
      norm_reward = tf.reduce_mean(self._reward_filter.transform(reward))
      # pylint: disable=g-long-lambda
      summary = tf.cond(self._should_log, lambda: tf.summary.merge([
        update_filters,
        self._observ_filter.summary(),
        self._reward_filter.summary(),
        tf.summary.scalar('memory_size', self._memory_index),
        tf.summary.histogram('normalized_observ', norm_observ),
        tf.summary.histogram('action', self._last_action),
        tf.summary.scalar('normalized_reward', norm_reward)]), str)
      return summary

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

      return _NetworkOutput(termination[:, 0, :], q_val[:, 0, :], options[:, 0, :, :], state[:, 0])


  def get_policy_over_options(self, network):
    self.probability_of_random_option = self._exploration_options.value()
    max_options = tf.cast(tf.argmax(network.q_val, 1), dtype=tf.int32)
    exp_options = tf.random_uniform(shape=[self._num_agents], minval=0, maxval=self._config.nb_options,
                              dtype=tf.int32)
    options = tf.map_fn(lambda i: tf.cond(self._random[i] > self.probability_of_random_option, lambda: max_options[i],
                   lambda: exp_options[i]), tf.range(0, self._num_agents))
    return options

  def get_action(self, network):
    self._current_option = tf.one_hot(self._current_option, self._nb_options, name="options_one_hot")
    self._current_option = tf.expand_dims(self._current_option, 2)
    self._current_option = tf.tile(self._current_option, [1, 1, self._action_size])
    self.action_probabilities = tf.reduce_sum(tf.multiply(network.options, self._current_option),
                                      reduction_indices=1, name="P_a")
    policy = tf.multinomial(tf.log(self.action_probabilities), 1)[:, 0]
    return policy


