from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

INPUT_SIZE = 1
TARGET_SIZE = 1
NUM_STEPS = 30
INIT_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 64
KEEP_PROB = 0.8
LSTM_SIZE = 128
NUM_LAYERS = 1
INIT_EPOCH = 5
MAX_EPOCH = 100
VECTOR_SIZE = 6

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

class LSTM:
    def __init__(self):
        tf.reset_default_graph()
        self.graph = tf.Graph()
    def define(self):
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, NUM_STEPS, VECTOR_SIZE])
            self.targets = tf.placeholder(tf.float32, [None, TARGET_SIZE])
            self.learning_rate = tf.placeholder(tf.float32, None)

            def _create_one_cell():
                return tf.contrib.rnn.LSTMCell(LSTM_SIZE)
                if KEEP_PROB < 1.0:
                    return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)

            self.cell = tf.contrib.rnn.MultiRNNCell([_create_one_cell() for _ in range(NUM_LAYERS)], state_is_tuple=True) if NUM_LAYERS > 1 else _create_one_cell()

            #input_weights = tf.Variable(tf.truncated_normal([VECTOR_SIZE]))
            #input_bias = tf.Variable(tf.constant(0.01, shape=[VECTOR_SIZE]))

            #input_layer = tf.matmul(inputs, input_weights) + input_bias

            self.val, _ = tf.nn.dynamic_rnn(self.cell, self.inputs, dtype=tf.float32)

            self.val = tf.transpose(self.val, [1,0,2])

            self.last = tf.gather(self.val, int(self.val.get_shape()[0]) - 1, name="last_lstm_output")

            self.weight = tf.Variable(tf.truncated_normal([LSTM_SIZE, VECTOR_SIZE]))
            self.bias = tf.Variable(tf.constant(0.01, shape=[VECTOR_SIZE]))
            self.prediction = tf.matmul(self.last, self.weight) + self.bias
            self.pred = tf.reduce_mean(self.prediction)

            self.loss = tf.reduce_mean(tf.square(self.prediction - self.targets))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.minimize = self.optimizer.minimize(self.loss)

    def train(self, train_inputs, train_targets):

        learning_rates_to_use = [INIT_LEARNING_RATE * (LEARNING_RATE_DECAY ** max(float(i + 1 - INIT_EPOCH), 0.0)) for i in range(MAX_EPOCH)]

        # Train graph
        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()

            for epoch_step in range(MAX_EPOCH):
                self.current_lr = learning_rates_to_use[epoch_step]
                for batch_X, batch_Y in zip(list(chunks(train_inputs, BATCH_SIZE)), list(chunks(train_targets, BATCH_SIZE))):
                    train_data_feed = {
                        self.inputs: batch_X,
                        self.targets: batch_Y,
                        self.learning_rate: self.current_lr
                    }
                    train_loss, _ = sess.run([self.loss, self.minimize], train_data_feed)
                    saver = tf.train.Saver()
                    saver.save(sess, "./model.ckpt")

    def test(self, test_inputs, test_targets):
        data_frame = [[],[],[]]

        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "./model.ckpt")
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./model_log", sess.graph)
            writer.add_graph(sess.graph) 
            i = 0
            for batch_X, batch_Y in zip(list(chunks(test_inputs, 1)), list(chunks(test_targets, 1))):
                test_data_feed = {self.inputs: batch_X, self.targets: batch_Y, self.learning_rate: self.current_lr}
                summary1, summary2, summary3 = sess.run([self.prediction, self.targets, self.pred], test_data_feed)
                i +=1
                data_frame[0].append(i)
                data_frame[1].append(np.ravel(summary3))
                data_frame[2].append(np.ravel(summary2))

        fig, ax = plt.subplots(figsize=(16,9))

        data_frame[2] = np.multiply(data_frame[2],1)
        data_frame[1] = np.multiply(data_frame[1],1)

        ax.plot(data_frame[2], label="target")
        ax.plot(data_frame[1], label="prediction")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        plt.show()
