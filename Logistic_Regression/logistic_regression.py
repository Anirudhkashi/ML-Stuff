import tensorflow as tf
import numpy as np

def logistic_regression(x_train, y_train, x_test, y_test, size, batch_size=20, epochs=2000, learning_rate=0.2):

	# Define input, output variables
	x = tf.placeholder(tf.float32, shape=[None, size])
	y_ = tf.placeholder(tf.float32, shape=[None, 1])

	# Define the parameter variable
	w = tf.Variable(tf.random_normal([size, 1],stddev=0.1))
	b = tf.Variable(tf.constant(0.0))

	# Output
	z = tf.add(tf.matmul(x, w), b)

	# Activation
	a = tf.sigmoid(z)

	# Define loss
	cost = tf.reduce_mean(-(y_*tf.log(a)+(1-y_)*tf.log(1-a)))

	# Train
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	# Add an op to initialize the variables.
	init_op = tf.global_variables_initializer()

	delta = tf.subtract(y_test, a)
	correct_prediction = tf.cast(tf.less(delta, tf.constant(0.5)), tf.float32)
	accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:

		# Initialise all the nodes
		sess.run(init_op)

		for epoch in range(epochs):

			# Fetch a random sample and do not replace it to not pick it again
			idx = np.random.choice(len(x_train),batch_size,replace=False)

			# Run the training step in batches
			_, l = sess.run([train_step, cost], feed_dict={x : x_train[idx], y_ : y_train[idx]})

			if epoch % 100 == 0:
				print "Number of epochs = " + str(epoch) + " ....... Loss = " + str(l)
		
		print float(accuracy.eval({x : x_test, y_ : y_test})) / len(x_test)

def load_data(filename):

	# Arrays to hold the labels and feature vectors.
	labels = []
	fvecs = []

	# Iterate over the rows, splitting the label from the features. Convert labels
	# to integers and features to floats.
	for line in file(filename):
		row = line.strip().split(",")
		labels.append(float(row[-1]))
		fvecs.append([float(x) for x in row[:-1]])

	# Convert the array of float arrays into a numpy float matrix.
	fvecs_np = np.matrix(fvecs).astype(np.float32)

	# Convert the array of int labels into a numpy array.
	labels = np.array(labels).astype(dtype=np.float32)

	# Return a pair of the feature matrix and the label matrix.
	return fvecs_np.reshape(fvecs_np.shape[0], 3), labels.reshape(labels.shape[0], 1)


def run():

	features, labels = load_data("./data/data.txt")
	index = int(len(features) * 0.7)
	x_train = features[:index]
	y_train = labels[:index]

	x_test = features[index:]
	y_test = labels[index:]

	num_features = features[0].shape[1]

	logistic_regression(x_train, y_train, x_test, y_test, num_features)

run()