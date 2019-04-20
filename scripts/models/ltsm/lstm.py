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
BATCH_SIZE =256
KEEP_PROB = 0.8
LSTM_SIZE = 512
NUM_LAYERS = 1
INIT_EPOCH = 5
MAX_EPOCH = 20
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
            print(NUM_LAYERS)
            self.inputs = tf.placeholder(tf.float32, [None, NUM_STEPS, VECTOR_SIZE])
            self.targets = tf.placeholder(tf.float32, [None, TARGET_SIZE])
            self.learning_rate = tf.placeholder(tf.float32, None)
            
            def _create_one_cell():
                lstm_cell = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
                #if KEEP_PROB < 1.0:
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
                #else:
                #    return lstm_cell
            #lstm_cell=tf.contrib.rnn.LSTMCell(LSTM_SIZE)
            #lstm_cell=tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB, seed=42)
            
            cell = tf.contrib.rnn.MultiRNNCell([_create_one_cell() for _ in range(NUM_LAYERS)], state_is_tuple=True) if NUM_LAYERS > 1 else _create_one_cell()
            #cell=tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS, state_is_tuple=True)
            
            #input_weights = tf.Variable(tf.truncated_normal([VECTOR_SIZE]))
            #input_bias = tf.Variable(tf.constant(0.01, shape=[VECTOR_SIZE]))
            
            #input_layer = tf.matmul(inputs, input_weights) + input_bias
            
            val, _ = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32)
            
            val = tf.transpose(val, [1,0,2])
            
            last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")
            
            weight = tf.Variable(tf.truncated_normal([LSTM_SIZE, 1]))
            bias = tf.Variable(tf.constant(0.01, shape=[1]))
            #self.prediction = tf.matmul(last, weight) + bias
            self.pred = tf.matmul(last, weight) + bias
            #self.pred = self.prediction
            
            #self.loss = tf.reduce_mean(tf.square(self.prediction - self.targets))
            #self.loss = tf.reduce_mean(tf.square(self.pred - self.targets))
            self.loss = tf.reduce_mean(tf.squared_difference(self.targets, self.pred))
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.minimize = optimizer.minimize(self.loss)
    
    def train(self, train_inputs, train_targets):
            
        learning_rates_to_use = [INIT_LEARNING_RATE * (LEARNING_RATE_DECAY ** max(float(i + 1 - INIT_EPOCH), 0.0)) for i in range(MAX_EPOCH)]

        # Train graph
        with tf.Session(graph=self.graph) as sess:
            #tf.global_variables_initializer().run()
            sess.run(tf.initialize_all_variables())
            
            train_pred_all=[]
            train_target_all=[]
            
            for epoch_step in range(MAX_EPOCH):
                print("Epoch:",epoch_step)
                self.current_lr = learning_rates_to_use[epoch_step]
                for batch_X, batch_Y in zip(list(chunks(train_inputs, BATCH_SIZE)), list(chunks(train_targets, BATCH_SIZE))):
                    #print(batch_X.shape, batch_Y.shape)
                    train_data_feed = {
                        self.inputs: batch_X,
                        self.targets: batch_Y,
                        self.learning_rate: self.current_lr
                    }
                    train_target, train_pred, train_loss, _ = sess.run([self.targets, self.pred, self.loss, self.minimize], train_data_feed)
                    print(train_loss)
                    if(epoch_step==MAX_EPOCH-1):
                        train_target_all.append(np.ravel(train_target))
                        train_pred_all.append(np.ravel(train_pred))
                    saver = tf.train.Saver()
                    saver.save(sess, "./model.ckpt")
            
            train_pred_all=np.concatenate(train_pred_all)
            train_target_all=np.concatenate(train_target_all)
            print(((train_pred_all-train_target_all)**2).mean())
            
            #fig, ax1 = plt.subplots(figsize=(16,9))
            #ax1.plot(train_target_all, label="target")
            #ax1.plot(train_pred_all, label="prediction")
            #ax1.set_xlabel('Date')
            #ax1.set_ylabel('Price')
            #ax1.set_title('Test Data')
            ##ax1.set_ylim(-1,2)
            #ax1.legend()
            #plt.show()

            
    def test(self, test_inputs, test_targets, train_inputs, train_targets):
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
                #summary1, summary2, summary3 = sess.run([self.prediction, self.targets, self.pred], test_data_feed)
                summary2, summary3 = sess.run([self.targets, self.pred], test_data_feed)
                i +=1
                data_frame[0].append(i)
                data_frame[1].append(np.ravel(summary3))
                data_frame[2].append(np.ravel(summary2))
            
        fig, (ax1, ax2) = plt.subplots(1, 2) #, figsize=(16,9))

        #data_frame[2] = np.multiply(data_frame[2],1)
        #data_frame[1] = np.multiply(data_frame[1],1)

        ax1.plot(data_frame[2], label="target")
        ax1.plot(data_frame[1], label="prediction")
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.set_title('Test Data')
        ax1.legend()
        
        mse_test=((np.array(data_frame[2])-np.array(data_frame[1]))**2).mean()
        print("Test MSE:", mse_test)
             

        # also plot training data
        data_frame = [[],[],[]]

        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "./model.ckpt")
            #merged_summary = tf.summary.merge_all()
            #writer = tf.summary.FileWriter("./model_log", sess.graph)
            #writer.add_graph(sess.graph) 
            i = 0
            for batch_X, batch_Y in zip(list(chunks(train_inputs, 1)), list(chunks(train_targets, 1))):
                test_data_feed = {self.inputs: batch_X, self.targets: batch_Y, self.learning_rate: self.current_lr}
                #summary1, summary2, summary3 = sess.run([self.prediction, self.targets, self.pred], test_data_feed)
                summary2, summary3 = sess.run([self.targets, self.pred], test_data_feed)
                i +=1
                data_frame[0].append(i)
                data_frame[1].append(np.ravel(summary3))
                data_frame[2].append(np.ravel(summary2))

        mse_train=((np.array(data_frame[2])-np.array(data_frame[1]))**2).mean()
        print("Train MSE:", mse_train)
        
        #data_frame[2] = np.multiply(data_frame[2],1)
        #data_frame[1] = np.multiply(data_frame[1],1)

        ax2.plot(data_frame[2], label="target")
        ax2.plot(data_frame[1], label="prediction")
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.set_title('Training Data')
        ax2.legend()
                  
                  

        plt.show()