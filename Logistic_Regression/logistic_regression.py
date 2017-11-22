import tensorflow as tf
import numpy as np

def train(x_train, y_train, batch_size=20, epochs=2000, learning_rate=0.2):

	# Define input, output variables
	x = tf.placeholder(tf.float32, shape=[None, size])
	y_ = tf.placeholder(tf.float32, shape=[None, 1])

	# Define the parameter variable
	w = tf.Variable(tf.truncated_normal_initializer(stddev=0.1))
	b = tf.Variable(tf.constant_initializer(0.0))

	# Output
	z = tf.add(tf.matmul(x, W), b)

	# Activation
	a = tf.sigmoid(z)

	# Define loss
	cost = tf.reduce_mean(-(y_*tf.log(a)+(1-y_)*tf.log(1-a)))

	# Train
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	# Add an op to initialize the variables.
	init_op = tf.global_variables_initializer()

	saver = tf.train.Saver()

	with tf.session() as sess:

		# Initialise all the nodes
		sess.run(init_op)

		for epoch in range(epochs):

			# Fetch a random sample and do not replace it to not pick it again
			idx = np.random.choice(len(X_train),batch_size,replace=False)

			# Run the training step in batches
			_, l = sess.run([train_step, cost], feed_dict={x : X_train[idx], y_ : y_train[idx]})

			if epoch % 100 == 0:
				print "Number of epochs = " + str(epoch) + " ....... Loss = " + str(l)

		saver.save(sess, "./model/model.ckpt")


def test(x_test, y_test, size):

	saver = tf.train.Saver()

	# Define input, output variables
	x = tf.placeholder(tf.float32, shape=[None, size])
	y_ = tf.placeholder(tf.float32, shape=[None, 1])

	correct_prediction = tf.equal(y_test, y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.session() as sess:
		saver.restore(sess, "./model/model.ckpt")
		result = accuracy.eval(feed_dict={x: x_test, y: y_test})

		print result


def load_data():

	filename_queue = tf.train.string_input_producer(["./data/data.txt"])

	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)

	# Default values, in case of empty columns. Also specifies the type of the
	# decoded result.

	x = tf.placeholder(tf.float32, shape=[None, 3])
	y = tf.placeholder(tf.float32, shape=[None, 1])

	record_defaults = [[1.0], [1.0], [1.0], [1.0]]
	col1, col2, col3, y = tf.decode_csv(
	    value, record_defaults=record_defaults)
	
	x = tf.pack([col1, col2, col3])

	with tf.Session() as sess:
	
		for i in data:
			print i.eval()

load_data()