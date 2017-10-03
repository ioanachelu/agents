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

def get_length_option(option_terminated):
  true_array = tf.ones_like(option_terminated)
  indices_option_terminated = tf.equal(option_terminated, true_array)
  first_option_terminated_index = indices_option_terminated[0]


sess = tf.InteractiveSession()
reward = tf.Variable([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
value = tf.Variable([[4.0, 5.0 , 6.0 , 7.0], [4.0, 5.0 , 6.0 , 7.0]])
length = tf.Variable([3, 2])
option_terminated = tf.Variable([[False, True, False, True], [False, False, True, True]])
sess.run(tf.global_variables_initializer())
real_length = get_length_option(option_terminated)
discount = 0.99

G = discounted_return_n_step(reward, length, discount, value)
print(G.eval())
print("avadakadavra")
