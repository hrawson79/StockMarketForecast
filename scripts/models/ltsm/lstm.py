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
VECTOR_SIZE = 7

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
            
            cell = tf.contrib.rnn.MultiRNNCell([_create_one_cell() for _ in range(NUM_LAYERS)], state_is_tuple=True) if NUM_LAYERS > 1 else _create_one_cell()
            
            input_weights = tf.Variable(tf.truncated_normal([VECTOR_SIZE]))
            input_bias = tf.Variable(tf.constant(0.01, shape=[VECTOR_SIZE]))
            
            #input_layer = tf.matmul(inputs, input_weights) + input_bias
            
            val, _ = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32)
            
            val = tf.transpose(val, [1,0,2])
            
            last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")
            
            weight = tf.Variable(tf.truncated_normal([LSTM_SIZE, INPUT_SIZE]))
            bias = tf.Variable(tf.constant(0.01, shape=[INPUT_SIZE]))
            prediction = tf.matmul(last, weight) + bias
            self.pred = prediction
            
            self.loss = tf.reduce_mean(tf.square(prediction - self.targets))
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.minimize = optimizer.minimize(self.loss)
    
    def train(self, train_inputs, train_targets):
        def chunks(l, n):
            # For item i in a range that is a length of l,
            for i in range(0, len(l), n):
                # Create an index range for l of n items:
                yield l[i:i+n]
            
        learning_rates_to_use = [INIT_LEARNING_RATE * (LEARNING_RATE_DECAY ** max(float(i + 1 - INIT_EPOCH), 0.0)) for i in range(MAX_EPOCH)]

        # Train graph
        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
        
            for epoch_step in range(MAX_EPOCH):
                current_lr = learning_rates_to_use[epoch_step]
                for batch_X, batch_Y in zip(list(chunks(train_inputs, BATCH_SIZE)), list(chunks(train_targets, BATCH_SIZE))):
                    train_data_feed = {
                        self.inputs: batch_X,
                        self.targets: batch_Y,
                        self.learning_rate: current_lr
                    }
                    train_loss, _ = sess.run([self.loss, self.minimize], train_data_feed)
                    saver = tf.train.Saver()
                    saver.save(sess, "model.ckpt")
