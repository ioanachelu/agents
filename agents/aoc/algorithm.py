
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
from agents.aoc.optimizers import huber_loss


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

    self._exploration_options = TFLinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                               self._config.initial_random_action_prob)
    self._exploration_policies = TFLinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                                self._config.initial_random_action_prob)
    # self._observ_filter = normalize.StreamingNormalize(
    #   self._batch_env.observ[0], center=True, scale=True, clip=5,
    #   name='normalize_observ')
    self._reward_filter = normalize.StreamingNormalize(
      self._batch_env.reward[0], center=False, scale=True, clip=10,
      name='normalize_reward')

    self._memory_index = tf.Variable(0, False)
    use_gpu = self._config.use_gpu and utility.available_gpus()
    with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
      self._network(
        tf.zeros_like(self._batch_env.observ)[:, None],
        tf.ones(len(self._batch_env)), reuse=None)
      cell = self._config.network(self._action_size)

      self._option_terminated = tf.Variable(
        np.zeros([self._num_agents], dtype=np.bool), False, name='option_terminated', dtype=tf.bool)
      self._frame_counter = tf.Variable(
        np.zeros([self._num_agents], dtype=np.int32), False, name='frame_counter', dtype=tf.int32)
      self._current_option = tf.Variable(
        np.zeros([self._num_agents], dtype=np.int32), False, name='current_option')
      self._random = tf.random_uniform(shape=[(self._config.num_agents)], minval=0., maxval=1., dtype=tf.float32)

      self._delib_cost = tf.Variable(np.asarray(self._num_agents * [self._config.delib_cost]),
                                      False, name="delib_cost", dtype=tf.float32)

      self._network_optimizer = self._config.network_optimizer(
        self._config.lr, name='network_optimizer')

    template = (
      self._batch_env.observ[0], self._batch_env.action[0], self._batch_env.action[0],
      self._batch_env.reward[0], self._option_terminated[0], self._batch_env.observ[0],
      self._option_terminated[0], tf.cast(self._batch_env.reward[0], dtype=tf.int32))
    self._memory = memory.EpisodeMemory(
      template, config.update_every, config.max_length, 'memory')

    with tf.variable_scope('aoc_temporary'):
      self._episodes = memory.EpisodeMemory(
        template, len(batch_env), config.max_length, 'episodes')
      self._last_state = utility.create_nested_vars(
        cell.zero_state(len(batch_env), tf.float32))
      # self._last_action = tf.Variable(
      #   tf.zeros_like(self._batch_env.action), False, name='last_action')

  def begin_episode(self, agent_indices):
    with tf.name_scope('begin_episode/'):
      reset_state = utility.reinit_nested_vars(self._last_state, agent_indices)
      reset_buffer = self._episodes.clear(agent_indices)
      variables = [self._frame_counter, self._current_option]
      reset_internal_vars = self.reinit_vars(variables)
      with tf.control_dependencies([reset_state, reset_buffer, reset_internal_vars]):
        return tf.constant('')

  def reinit_vars(self, variables):
    if isinstance(variables, (tuple, list)):
      return tf.group(*[
        self.reinit_vars(variable) for variable in variables])
    return tf.group(variables.assign(tf.zeros_like(variables)),
      self._option_terminated.assign(np.asarray(self._num_agents * [False])),
      self._delib_cost.assign(np.asarray(self._num_agents * [self._config.delib_cost])))


  def perform(self, observ):
    with tf.name_scope('perform/'):
      # observ = self._observ_filter.transform(observ)
      self.network = network = self._network(
        observ[:, None], tf.ones(observ.shape[0]), self._last_state)

      next_options = self.get_policy_over_options(network)
      self._current_option = tf.where(self._option_terminated, next_options, self._current_option)

      # self._current_option[self._option_terminated] = self.get_policy_over_options(network)[self._option_terminated]

      action = self.get_action(network)

    return tf.cast(action, dtype=tf.int32) , tf.constant('')

  def experience(self, observ, action, reward, done, nextob, agent_indices):
    with tf.name_scope('experience/'):
      return tf.cond(
        self._is_training,
        lambda: self._define_experience(observ, action, reward, done, nextob, agent_indices), str)

  def _define_experience(self, observ, action, reward, done, nextob, agent_indices):
    """Implement the branch of experience() entered during training."""
    update_filters = tf.summary.merge([
      # self._observ_filter.update(observ),
      self._reward_filter.update(reward)])

    # norm_observ = self._observ_filter.transform(observ)
    norm_reward = self._reward_filter.transform(reward)

    with tf.control_dependencies([update_filters]):
      # if self._config.train_on_agent_action:
      #   # NOTE: Doesn't seem to change much.
      #   action = self._last_action

      increment_frame_counter = self._frame_counter.assign_add(tf.ones(self._num_agents, tf.int32))

      float_option_terminated = tf.cast(self._option_terminated, dtype=tf.float32)

      new_reward = norm_reward - \
                   float_option_terminated * self._delib_cost * tf.cast(self._frame_counter, dtype=tf.float32)

      self._option_terminated = self.get_termination()

      batch = observ, self._current_option, action, new_reward, done, nextob, self._option_terminated, agent_indices
      append = self._episodes.append(batch, tf.range(len(self._batch_env)))
    with tf.control_dependencies([append]):
      # pylint: disable=g-long-lambda
      summary = tf.cond(self._should_log, lambda: tf.summary.merge([
        update_filters,
        # self._observ_filter.summary(),
        self._reward_filter.summary(),
        tf.summary.scalar('memory_size', self._memory_index),
        # tf.summary.image(observ),
        tf.summary.histogram('observ', observ),
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

      return _NetworkOutput(termination, q_val, options, state)


  def get_policy_over_options(self, network):
    self.probability_of_random_option = self._exploration_options.value(self._step)
    max_options = tf.cast(tf.argmax(network.q_val[:, 0,:], 1), dtype=tf.int32)
    exp_options = tf.random_uniform(shape=[self._num_agents], minval=0, maxval=self._config.nb_options,
                              dtype=tf.int32)
    options = tf.where(self._random > self.probability_of_random_option, max_options, exp_options)
    return options

  def get_action(self, network):
    current_option_option_one_hot = tf.one_hot(self._current_option, self._nb_options, name="options_one_hot")
    current_option_option_one_hot = current_option_option_one_hot[:, :, None]
    current_option_option_one_hot = tf.tile(current_option_option_one_hot, [1, 1, self._action_size])
    self.action_probabilities = tf.reduce_sum(tf.multiply(network.options[:, 0,:], current_option_option_one_hot),
                                      reduction_indices=1, name="P_a")
    policy = tf.multinomial(tf.log(self.action_probabilities), 1)[:, 0]
    return policy

  def get_termination(self):
    current_option_option_one_hot = tf.one_hot(self._current_option, self._nb_options, name="options_one_hot")
    termination_probabilities = tf.reduce_sum(tf.multiply(self.network.termination[:, 0,:], current_option_option_one_hot),
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
      (observ, option, action, reward, done, nextob, option_terminated, agent_index), length = data
      with tf.control_dependencies([tf.assert_greater(length, 0)]):
        length = tf.identity(length)

      network_summary = self._update_network(
        observ, option, action, reward, done, nextob, option_terminated, agent_index, length)

      with tf.control_dependencies([network_summary]):
        clear_memory = tf.group(
            self._memory.clear(), self._memory_index.assign(0))
      with tf.control_dependencies([clear_memory]):
        weight_summary = utility.variable_summaries(
            tf.trainable_variables(), self._config.weight_summaries)
        return tf.summary.merge([network_summary, weight_summary])

  def _update_network(self, observ, option, action, reward, done, nextob, option_terminated, agent_index, length):
    self.probability_of_random_option = self._exploration_options.value(self._step)
    with tf.name_scope('update_network'):
      # add delib if  option termination because it isn't part of V
      delib = self._delib_cost * tf.cast(self._frame_counter > 1, dtype=np.float32)
      network = self._network(observ, length)
      network_next = self._network(nextob, length)
      # raw_v = tf.reduce_sum(tf.multiply(self.get_V(network), tf.one_hot(length, self._config.max_length)), axis=1)
      v = self.get_V(network_next) - tf.tile(delib[:, None], [1, self._config.max_length])
      # q = tf.reduce_sum(tf.multiply(self.get_Q(network, option), tf.one_hot(length, self._config.max_length)), axis=1)
      q = self.get_Q(network_next, option)
      new_v = tf.where(option_terminated, v, q)
      R = tf.cast(tf.logical_not(done), dtype=tf.float32) * new_v

      advantage = utility.lambda_advantage(reward, R, length, self._config.discount)

      mean, variance = tf.nn.moments(advantage, axes=[0, 1], keep_dims=True)
      advantage = (advantage - mean) / (tf.sqrt(variance) + 1e-8)

      advantage = tf.Print(
        advantage, [tf.reduce_mean(advantage)],
        'normalized advantage: ')

      q_opt = self.get_Q(network, option)
      v = self.get_V(network)
      intra_option_policy = self.get_intra_option_policy(network, option)
      responsible_outputs = self.get_responsible_outputs(intra_option_policy, action)
      o_termination = self.get_option_termination(network, option)
      with tf.name_scope('critic_loss'):
        td_error = advantage - q_opt
        critic_loss = tf.reduce_mean(self._config.critic_coef * 0.5 * tf.square((td_error)))
      with tf.name_scope('termination_loss'):
        term_loss = -tf.reduce_mean(o_termination * (tf.stop_gradient(q_opt) - tf.stop_gradient(v) +
                                                          tf.tile(delib[:, None],
                                                                  [1, self._config.max_length])))
      with tf.name_scope('entropy_loss'):
        entropy_loss = self._config.entropy_coef * tf.reduce_mean(tf.reduce_sum(intra_option_policy *
                                                                                tf.log(intra_option_policy +
                                                                                       1e-7), axis=2))
      with tf.name_scope('policy_loss'):
        policy_loss = -tf.reduce_sum(
                    tf.log(responsible_outputs + 1e-7) * advantage)

      total_loss = policy_loss + entropy_loss + critic_loss + term_loss

      gradients, variables = (
        zip(*self._network_optimizer.compute_gradients(total_loss)))
      optimize = self._network_optimizer.apply_gradients(
        zip(gradients, variables))
      summary = tf.summary.merge([
        tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
        utility.gradient_summaries(
          zip(gradients, variables))])
      with tf.control_dependencies([optimize]):
        print_loss = tf.Print(0, [total_loss], 'network loss: ')

      with tf.control_dependencies([total_loss, print_loss]):
        return summary

  def get_intra_option_policy(self, network, option):
    current_option_option_one_hot = tf.one_hot(option, self._nb_options, name="options_one_hot")
    current_option_option_one_hot = tf.tile(current_option_option_one_hot[..., None], [1, 1, 1, self._action_size])
    action_probabilities = tf.reduce_sum(tf.multiply(network.options, current_option_option_one_hot),
                                              reduction_indices=2, name="P_a")
    return action_probabilities

  def get_option_termination(self, network, current_option):
    current_option_option_one_hot = tf.one_hot(current_option, self._nb_options, name="options_one_hot")
    o_terminations = tf.reduce_sum(tf.multiply(network.termination, current_option_option_one_hot),
                             reduction_indices=2, name="O_Terminations")
    return o_terminations

  def get_responsible_outputs(self, policy, action):
    actions_onehot = tf.one_hot(action, self._action_size, dtype=tf.float32,
                                     name="Actions_Onehot")
    responsible_outputs = tf.reduce_sum(policy * actions_onehot, [2])
    return responsible_outputs

  def get_V(self, network):
    q_val = network.q_val
    v = tf.reduce_max(q_val, axis=2) * (1 - self.probability_of_random_option) +\
        self.probability_of_random_option * tf.reduce_mean(q_val, axis=2)
    return v

  def get_Q(self, network, current_option):
    current_option_option_one_hot = tf.one_hot(current_option, self._nb_options, name="options_one_hot")
    q_values = tf.reduce_sum(tf.multiply(network.q_val, current_option_option_one_hot),
                                              reduction_indices=2, name="Values_Q")
    return q_values
