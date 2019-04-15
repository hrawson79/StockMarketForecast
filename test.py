import scripts.data.model_data as model_data
import scripts.data.stock_data as stock_data

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    # Stock symbols to be downloaded
    tickers = ['^GSPC']

    # Date range of data to be downloaded
    start_date = '2018-06-02'
    end_date = '2019-03-15'

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
    MAX_EPOCH = 50
    VECTOR_SIZE = 6

    stock_1 = stock_data.stock(tickers, start_date, end_date)
    data_1 = model_data.model_data(stock_1, INPUT_SIZE, NUM_STEPS)


    # Initialize graph
    tf.reset_default_graph()
    lstm_graph = tf.Graph()

    # Define the graph
    with lstm_graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, NUM_STEPS, VECTOR_SIZE])
        targets = tf.placeholder(tf.float32, [None, TARGET_SIZE])
        learning_rate = tf.placeholder(tf.float32, None)
        
        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
            if KEEP_PROB < 1.0:
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
            else:
                return lstm_cell
        
        cell = tf.contrib.rnn.MultiRNNCell([_create_one_cell() for _ in range(NUM_LAYERS)], state_is_tuple=True) if NUM_LAYERS > 1 else _create_one_cell()
        
        # input_weights = tf.Variable(tf.truncated_normal([VECTOR_SIZE]))
        # input_bias = tf.Variable(tf.constant(0.01, shape=[VECTOR_SIZE]))
        
        #input_layer = tf.matmul(inputs, input_weights) + input_bias
        
        val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        
        val = tf.transpose(val, [1,0,2])
        
        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")
        
        weight = tf.Variable(tf.truncated_normal([LSTM_SIZE, INPUT_SIZE]))
        bias = tf.Variable(tf.constant(0.01, shape=[INPUT_SIZE]))
        prediction = tf.matmul(last, weight) + bias
        pred = prediction
        
        loss = tf.reduce_mean(tf.square(prediction - targets))
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        minimize = optimizer.minimize(loss)

    # Create a function called "chunks" with two arguments, l and n:
    def chunks(l, n):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i+n]
        
    learning_rates_to_use = [INIT_LEARNING_RATE * (LEARNING_RATE_DECAY ** max(float(i + 1 - INIT_EPOCH), 0.0)) for i in range(MAX_EPOCH)]

    # Train graph
    with tf.Session(graph=lstm_graph) as sess:
        tf.global_variables_initializer().run()
    
        for epoch_step in range(MAX_EPOCH):
            current_lr = learning_rates_to_use[epoch_step]
            for batch_X, batch_Y in zip(list(chunks(data_1.train_inputs_, BATCH_SIZE)), list(chunks(data_1.train_targets_, BATCH_SIZE))):
                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_Y,
                    learning_rate: current_lr
                }
                train_loss, _ = sess.run([loss, minimize], train_data_feed)
                saver = tf.train.Saver()
                saver.save(sess, "./model.ckpt")

    data_frame = [[],[],[]]

    with tf.Session(graph=lstm_graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./model.ckpt")
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./model_log", sess.graph)
        writer.add_graph(sess.graph) 
        i = 0
        for batch_X, batch_Y in zip(list(chunks(data_1.test_inputs_, 1)), list(chunks(data_1.test_targets_, 1))):
            test_data_feed = {inputs: batch_X, targets: batch_Y, learning_rate: current_lr}
            summary1, summary2, summary3 = sess.run([prediction, targets, pred], test_data_feed)
            i +=1
            data_frame[0].append(i)
            data_frame[1].append(np.ravel(summary3))
            data_frame[2].append(np.ravel(summary2))
        
    fig, ax = plt.subplots(figsize=(16,9))

    data_frame[2] = np.multiply(data_frame[2],1)
    data_frame[1] = np.multiply(data_frame[1],1)

    ax.plot(data_1.test_dates_, data_frame[2], label="target")
    ax.plot(data_1.test_dates_, data_frame[1], label="prediction")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    plt.show()

    print("End")

