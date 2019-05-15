%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')
x_train, y_train = mnist.train.images, mnist.train.labels

learning_rate = 0.01
n_epochs = 150
batch_size = 100
feature_size = 784
embed_size = 64

n_train = len(x_train)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(1000)
train_data = train_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                          train_data.output_shapes)

features, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)

w1 = tf.Variable(tf.random_normal(shape=[feature_size, 393], stddev=0.01), name="weights1")
b1 = tf.Variable(tf.zeros([1, 393]), name="bias1")
h1 = tf.add(tf.matmul(tf.cast(features, tf.float32), w1), b1)
h1 = tf.layers.batch_normalization(h1, training=True)
h1 = tf.nn.relu(h1)

w2 = tf.Variable(tf.random_normal(shape=[393, embed_size], stddev=0.01), name="weights2")
b2 = tf.Variable(tf.zeros([1, embed_size]), name="bias2")
h2 = tf.add(tf.matmul(tf.cast(h1, tf.float32), w2), b2)
h2 = tf.layers.batch_normalization(h2, training=True)
emb = tf.nn.relu(h2)

w3 = tf.Variable(tf.random_normal(shape=[embed_size, 393], stddev=0.01), name="weights2")
b3 = tf.Variable(tf.zeros([1, 393]), name="bias3")
h3 = tf.add(tf.matmul(tf.cast(emb, tf.float32), w3), b3)
h3 = tf.layers.batch_normalization(h3, training=True)
h3 = tf.nn.relu(h3)

w4 = tf.Variable(tf.random_normal(shape=[393, feature_size], stddev=0.01), name="weights2")
b4 = tf.Variable(tf.zeros([1, feature_size]), name="bias4")
h4 = tf.add(tf.matmul(tf.cast(h3, tf.float32), w4), b4)
h4 = tf.layers.batch_normalization(h4, training=True)
output = tf.nn.relu(h4)

loss = tf.reduce_mean(tf.square(output - features))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    batch_losses = []
    for i in range(n_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                batch_losses.append(l)
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        if i % 10 == 0:
            print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    saver.save(sess, "/tmp/model.ckpt")

    indices = [3, 2, 1, 18, 4, 8, 11, 0, 61, 9]
    results = sess.run(output, feed_dict={features:mnist.test.images[indices]})

# Plot the digits. The original images on the top and the reconstructed at the
# bottom.
f,a = plt.subplots(2, 10, figsize=(20, 4))
for i, _ in enumerate(indices):
    a[0][i].imshow(np.reshape(mnist.test.images[indices[i]], (28,28)))
    a[1][i].imshow(np.reshape(results[i], (28,28)))