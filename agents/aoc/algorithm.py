
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections

from agents.aoc import memory
from agents.aoc import normalize
from agents.aoc import utility
import numpy as np

from agents.aoc.schedules import LinearSchedule, TFLinearSchedule


_NetworkOutput = collections.namedtuple(
    'NetworkOutput', 'termination, q_val, options, state')

class AOCAlgorithm(object):

  def __init__(self, batch_env, step, is_training, should_log, config):
    self._batch_env = batch_env
    self._step = step
    self._is_training = is_training
    self._should_log = should_log
    self.t_step = 0

    self._config = config
    self._frame_counter = 0
    self._nb_options = config.nb_options
    self._action_size = self._batch_env._batch_env._envs[0]._action_space.n
    self._num_agents = self._config.num_agents

    self._exploration_options = TFLinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                               self._config.initial_random_action_prob)
    self._exploration_policies = TFLinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                                self._config.initial_random_action_prob)
    self._observ_filter = normalize.StreamingNormalize(
      self._batch_env.observ[0], center=True, scale=True, clip=5,
      name='normalize_observ')
    self._reward_filter = normalize.StreamingNormalize(
      self._batch_env.reward[0], center=False, scale=True, clip=10,
      name='normalize_reward')
    # Memory stores tuple of observ, action, mean, logstd, reward.
    template = (
      self._batch_env.observ[0], self._batch_env.action[0], self._batch_env.action[0],
      self._batch_env.reward[0], self._option_terminated[0], self._batch_env.observ[0],
      self._option_terminated[0])
    self._memory = memory.EpisodeMemory(
      template, config.update_every, config.max_length, 'memory')
    self._memory_index = tf.Variable(0, False)
    use_gpu = self._config.use_gpu and utility.available_gpus()
    with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
      self._network(
        tf.zeros_like(self._batch_env.observ)[:, None],
        tf.ones(len(self._batch_env)), reuse=None)
      cell = self._config.network(self._batch_env._batch_env._envs[0]._action_space.n)

      self._option_terminated = tf.Variable(
        tf.zeros([self._num_agents], dtype=tf.bool), False, name='option_terminated')
      self._frame_counter = tf.Variable(
        tf.zeros([self._num_agents], dtype=tf.int32), False, name='frame_counter')
      self._current_option = tf.Variable(
        tf.zeros([self._num_agents], dtype=tf.int32), False, name='current_option')
      self._random = tf.random_uniform(shape=[(self._config.num_agents)], minval=0., maxval=1., dtype=tf.float32)

      self._delib_cost = tf.Variable(tf.convert_to_tensor(np.asarray(self._num_agents * [self._config.delib_cost])),
                                      False, name="delib_cost")
      self._t_counter = tf.Variable(0, dtype=tf.int32)

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
      variables = [self._option_terminated, self._frame_counter, self._current_option]
      reset_internal_vars = self.reinit_vars(variables)
      with tf.control_dependencies([reset_state, reset_buffer, reset_internal_vars]):
        return tf.constant('')

  def reinit_vars(self, variables):
    variables.assign(tf.zeros_like(variables))
    self._delib_cost.assign(np.asarray(self._num_agents * [self._config.delib_cost]))


  def perform(self, observ):
    self.t_step += 1
    with tf.name_scope('perform/'):
      observ = self._observ_filter.transform(observ)
      self.network = network = self._network(
        observ[:, None], tf.ones(observ.shape[0]), self._last_state)

      next_options = self.get_policy_over_options(network)
      self._current_option = tf.map_fn(lambda i: tf.cond(self._option_terminated[i], lambda: next_options[i],
                                                         lambda: self._current_option[i]),
                                                         tf.range(self._num_agents))
      # self._current_option[self._option_terminated] = self.get_policy_over_options(network)[self._option_terminated]

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

    norm_observ = self._observ_filter.transform(observ)
    norm_reward = self._reward_filter.transform(reward)

    with tf.control_dependencies([update_filters]):
      # if self._config.train_on_agent_action:
      #   # NOTE: Doesn't seem to change much.
      #   action = self._last_action
      self._frame_counter += 1
      new_reward = norm_reward - \
                        tf.cast(self._option_terminated, dtype=tf.float32) * self._delib_cost * self._frame_counter

      self._option_terminated = self.get_termination()

      batch = observ, self._current_option, action, new_reward, done, nextob, self._option_terminated
      append = self._episodes.append(batch, tf.range(len(self._batch_env)))
    with tf.control_dependencies([append]):
      # pylint: disable=g-long-lambda
      summary = tf.cond(self._should_log, lambda: tf.summary.merge([
        update_filters,
        self._observ_filter.summary(),
        self._reward_filter.summary(),
        tf.summary.scalar('memory_size', self._memory_index),
        # tf.summary.image(observ),
        tf.summary.histogram('normalized_observ', norm_observ),
        tf.summary.scalar('action', action),
        tf.summary.scalar('option', self._current_option),
        tf.summary.scalar('option_terminated', tf.cast(self._option_terminated, dtype=tf.int32)),
        tf.summary.scalar('done', tf.cast(done, dtype=tf.int32)),
        tf.summary.scalar('normalized_reward', norm_reward)]), str)
      return summary

  def end_episode(self, agent_indices):
    with tf.name_scope('end_episode/'):
      return tf.cond(
          self._is_training,
          lambda: self._define_end_episode(agent_indices), str)

  def _define_end_episode(self, agent_indices):
    episodes, length = self._episodes.data(agent_indices)
    space_left = self._config.update_every - self._memory_index
    use_episodes = tf.range(tf.minimum(
        tf.shape(agent_indices)[0], space_left))
    episodes = [tf.gather(elem, use_episodes) for elem in episodes]
    append = self._memory.replace(
        episodes, tf.gather(length, use_episodes),
        use_episodes + self._memory_index)
    with tf.control_dependencies([append]):
      inc_index = self._memory_index.assign_add(tf.shape(use_episodes)[0])
    with tf.control_dependencies([inc_index]):
      memory_full = self._memory_index >= self._config.update_every
      return tf.cond(memory_full, self._training, str)

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
    self.probability_of_random_option = self._exploration_options.value(self.t_step)
    max_options = tf.cast(tf.argmax(network.q_val, 1), dtype=tf.int32)
    exp_options = tf.random_uniform(shape=[self._num_agents], minval=0, maxval=self._config.nb_options,
                              dtype=tf.int32)
    options = tf.map_fn(lambda i: tf.cond(self._random[i] > self.probability_of_random_option, lambda: max_options[i],
                   lambda: exp_options[i]), tf.range(0, self._num_agents))
    return options

  def get_action(self, network):
    current_option_option_one_hot = tf.one_hot(self._current_option, self._nb_options, name="options_one_hot")
    current_option_option_one_hot = tf.expand_dims(current_option_option_one_hot, 2)
    current_option_option_one_hot = tf.tile(current_option_option_one_hot, [1, 1, self._action_size])
    self.action_probabilities = tf.reduce_sum(tf.multiply(network.options, current_option_option_one_hot),
                                      reduction_indices=1, name="P_a")
    policy = tf.multinomial(tf.log(self.action_probabilities), 1)[:, 0]
    return policy

  def get_termination(self):
    current_option_option_one_hot = tf.one_hot(self._current_option, self._nb_options, name="options_one_hot")
    termination_probabilities = tf.reduce_sum(tf.multiply(self.network.termination, current_option_option_one_hot),
                  reduction_indices=1, name="P_term")
    terminated = termination_probabilities > self._random
    return terminated

  def _training(self):
    """Perform one training iterations of both policy and value baseline.

    Training on the episodes collected in the memory. Reset the memory
    afterwards. Always returns a summary string.

    Returns:
      Summary tensor.
    """
    with tf.name_scope('training'):
      assert_full = tf.assert_equal(
          self._memory_index, self._config.update_every)
      with tf.control_dependencies([assert_full]):
        data = self._memory.data()
      (observ, option, action, reward, done, nextob, option_terminated), length = data
      with tf.control_dependencies([tf.assert_greater(length, 0)]):
        length = tf.identity(length)

      network_summary = self._update_network(
        observ, option, action, reward, done, nextob, option_terminated, length)
      with tf.control_dependencies([network_summary]):
        clear_memory = tf.group(
            self._memory.clear(), self._memory_index.assign(0))
      with tf.control_dependencies([clear_memory]):
        weight_summary = utility.variable_summaries(
            tf.trainable_variables(), self._config.weight_summaries)
        return tf.summary.merge([network_summary, weight_summary])

  def _update_network(self, observ, option, action, reward, done, nextob, option_terminated, length):
    """Perform one update step on the network.

    The advantage is computed once at the beginning and shared across
    iterations. We need to decide for the summary of one iteration, and thus
    choose the one after half of the iterations.

    Args:
      observ: Sequences of observations.
      action: Sequences of actions.
      old_mean: Sequences of action means of the behavioral policy.
      old_logstd: Sequences of action log stddevs of the behavioral policy.
      reward: Sequences of rewards.
      length: Batch of sequence lengths.

    Returns:
      Summary tensor.
    """
    self.probability_of_random_option = self._exploration_options.value(self.t_step)
    with tf.name_scope('update_network'):
      # add delib if  option termination because it isn't part of V
      delib = self._delib_cost * np.asarray(self._frame_counter > 1, dtype=np.float32)
      network = self._network(observ, length)
      v = self.get_V(network) - delib
      q = self.get_Q(network, option)
      new_v = tf.map_fn(lambda i: tf.cond(option_terminated[i], lambda: v[i], lambda: q[i]),
                        tf.range(self._config.update_every))


      utility.fixed_step_return(reward, value, length, discount, window)

      return_ = utility.discounted_return(
        reward, length, self._config.discount)
      value = self._network(observ, length).value
      if self._config.gae_lambda:
        advantage = utility.lambda_return(
          reward, value, length, self._config.discount,
          self._config.gae_lambda)
      else:
        advantage = return_ - value
      mean, variance = tf.nn.moments(advantage, axes=[0, 1], keep_dims=True)
      advantage = (advantage - mean) / (tf.sqrt(variance) + 1e-8)
      advantage = tf.Print(
        advantage, [tf.reduce_mean(return_), tf.reduce_mean(value)],
        'return and value: ')
      advantage = tf.Print(
        advantage, [tf.reduce_mean(advantage)],
        'normalized advantage: ')
      # pylint: disable=g-long-lambda
      loss, summary = tf.scan(
        lambda _1, _2: self._update_policy_step(
          observ, action, old_mean, old_logstd, advantage, length),
        tf.range(self._config.update_epochs_policy),
        [0., ''], parallel_iterations=1)
      print_loss = tf.Print(0, [tf.reduce_mean(loss)], 'policy loss: ')
      with tf.control_dependencies([loss, print_loss]):
        return summary[self._config.update_epochs_policy // 2]


  def get_V(self, network):
    q_val = network.q_val
    v = tf.reduce_max(q_val, axis=1) * (1 - self.probability_of_random_option) +\
        self.probability_of_random_option * tf.reduce_mean(q_val, axis=1)
    return v

  def get_Q(self, network, current_option):
    current_option_option_one_hot = tf.one_hot(current_option, self._nb_options, name="options_one_hot")
    q_values = tf.reduce_sum(tf.multiply(network.q_val, current_option_option_one_hot),
                                              reduction_indices=1, name="Values_Q")
    return q_values
