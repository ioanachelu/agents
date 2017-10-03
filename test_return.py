import tensorflow as tf

def discounted_return_n_step(reward, length, discount, values):
  """Discounted n-step returns Monte-Carlo returns."""
  timestep = tf.tile(tf.range(reward.shape[1].value)[None, ...], [reward.shape[0].value, 1])
  # mask = tf.cast(timestep < length[:, None], tf.float32)
  indices = tf.where(timestep < length[:, None])
  value = tf.gather(tf.gather_nd(values, indices), length - 1)
  augmented_rewards = [tf.concat([rewards, value], 0) for rewards, value in
                       zip(tf.split(tf.gather_nd(reward, indices), length),
                           tf.split(tf.transpose(value).eval(), tf.ones_like(length)))]
  lengths = tf.split(length, tf.ones_like(length))
  padded_augmented_rewards = tf.stack([tf.pad(a_r, [[0, (reward.shape[1].value - length[0] - 1)]]) for a_r, length in zip(augmented_rewards, lengths)], 0)
  return_ = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur + discount * agg,
      tf.transpose(tf.reverse(padded_augmented_rewards, [1]), [1, 0]),
      tf.zeros_like(reward[:, -1]), 1, False), [1, 0]), [1])
  return tf.check_numerics(tf.stop_gradient(return_), 'return')

def get_length_option(option_terminated, length):
  real_length_option_terminated = tf.transpose(tf.map_fn(lambda f: tf.cond(tf.cast(tf.shape(tf.where(tf.cast(f, dtype=tf.bool)))[0], tf.bool),
                                                                           lambda: tf.where(tf.cast(f, dtype=tf.bool))[
                                                                             0],
                                                         lambda: tf.cast(option_terminated.shape[1].value, tf.int64)[..., None]),
                                                         tf.cast(option_terminated, dtype=tf.int64)), [1, 0])[0]
  real_length_option_terminated = tf.cast(real_length_option_terminated, dtype=tf.int32)
  new_length = tf.where((real_length_option_terminated < length), real_length_option_terminated, length)

  return new_length

def fixed_step_return(reward, value, length, discount, window):
  """N-step discounted return."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  return_ = tf.zeros_like(reward)
  for _ in range(window):
    return_ += reward
    reward = discount * tf.concat(
        [reward[:, 1:], tf.zeros_like(reward[:, -1:])], 1)
  return_ += discount ** window * tf.concat(
      [value[:, window:], tf.zeros_like(value[:, -window:]), 1])
  return tf.check_numerics(tf.stop_gradient(mask * return_), 'return')

def lambda_return(reward, value, length, discount, lambda_):
  """TD-lambda returns."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  sequence = mask * reward + discount * value * (1 - lambda_)
  discount = mask * discount * lambda_
  sequence = tf.stack([sequence, discount], 2)
  return_ = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur[0] + cur[1] * agg,
      tf.transpose(tf.reverse(sequence, [1]), [1, 2, 0]),
      tf.zeros_like(value[:, -1]), 1, False), [1, 0]), [1])
  return tf.check_numerics(tf.stop_gradient(return_), 'return')


sess = tf.InteractiveSession()
reward = tf.Variable([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
value = tf.Variable([[4.0, 5.0 , 6.0 , 7.0], [4.0, 5.0 , 6.0 , 7.0]])
length = tf.Variable([0, 0])
option_terminated = tf.Variable([[False, False, False, False], [False, False, False, False]])
sess.run(tf.global_variables_initializer())
real_length = get_length_option(option_terminated, length)
discount = 0.99

G = lambda_return(reward, value, real_length, discount, 0.98)
G1 = discounted_return_n_step(reward, real_length, discount, value)
print(G.eval())
print(G1.eval())
print("avadakadavra")
