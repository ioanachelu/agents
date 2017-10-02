import tensorflow as tf

def discounted_return(reward, length, discount, values):
  """Discounted Monte-Carlo returns."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  indices = tf.where(timestep[None, :] < length[:, None])
  value = tf.gather_nd(values, indices)[-1]
  return_ = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur + discount * agg,
      tf.transpose(tf.reverse(mask * reward, [1]), [1, 0]),
      value[None, ...], 1, False), [1, 0]), [1])
  return tf.check_numerics(tf.stop_gradient(return_), 'return')

sess = tf.InteractiveSession()
reward = tf.Variable([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
value = tf.Variable([[4.0, 5.0 , 6.0 , 7.0], [4.0, 5.0 , 6.0 , 7.0]])
length = tf.Variable([3, 2])
discount = 0.99
sess.run(tf.global_variables_initializer())
G = discounted_return(reward, length, discount, value)
print(G.eval())
print("avadakadavra")
