from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

# Function to load data for stocks defined in tickers array
def LoadStockDataForTickers(tickers, start_date, end_date, source='yahoo'):
  # Use pandas_reader.data.DataReader to load the desired data
  panel_data = data.DataReader(tickers, source, start_date, end_date)
  
  close = panel_data

  # Get all of the weekdays within the range specified above
  all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

  # Reindex close using all_weekdays as new index
  close = close.reindex(all_weekdays)

  # Fill in missing dates
  close = close.fillna(method='ffill')
  
  return close

# Plot a specified column of specified stock
def PlotColumnForTicker(data, column, ticker):
  data_frame = data[column]
  
  data_frame = data_frame.loc[:, ticker]

  fig, ax = plt.subplots(figsize=(16,9))

  ax.plot(data_frame.index, data_frame, label=ticker)

  ax.set_xlabel('Date')
  ax.set_ylabel('Adjusted closing price ($)')
  ax.legend()

# constants
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


# Stock symbols to be downloaded
tickers = ['^GSPC']

# Date range of data to be downloaded
start_date = '2017-02-02'
end_date = '2019-03-15'

# Download the data
sap_data = LoadStockDataForTickers(tickers, start_date, end_date)

# Add a column and compute the 20 day moving average
sap_data['20 Day Moving'] = (sap_data['Close'].rolling(window=20).mean())
#sap_data['100 Day Moving'] = (sap_data['Close'].rolling(window=100).mean())

# Drop first 20 rows because moving avearge will be NaN
sap_data = sap_data.iloc[19:]

# Close data to use as targets
close_data = sap_data['Close']

# Store the dates for plotting
dates = sap_data.index.values

# Show some of the data
sap_data.head()

# Format data for input
seq = [np.array(close_data[i * INPUT_SIZE: (i + 1) * INPUT_SIZE]) 
       for i in range(len(close_data) // INPUT_SIZE)]
seq2 = [np.array(sap_data[i * INPUT_SIZE: (i + 1) * INPUT_SIZE]) 
       for i in range(len(sap_data) // INPUT_SIZE)]
seq3 = [np.array(dates[i * INPUT_SIZE: (i + 1) * INPUT_SIZE]) 
       for i in range(len(dates) // INPUT_SIZE)]

# Split into groups of `num_steps`
X = np.array([seq2[i: i + NUM_STEPS] / seq2[i-1] for i in range(len(seq2) - NUM_STEPS)])
Y = np.array([seq[i + NUM_STEPS] / seq[i-1] for i in range(len(seq) - NUM_STEPS)])
X = X.reshape(len(X), NUM_STEPS, VECTOR_SIZE)
Y = Y.reshape(len(Y), INPUT_SIZE)
# Split data into train and test sets
X_TRAIN = X[:int((len(X)*.8))]
X_TEST = X[int((len(X)*.8))+1:]
Y_TRAIN = Y[:int((len(Y)*.8))]
Y_TEST = Y[int((len(Y)*.8))+1:]

D = np.array([seq3[i + NUM_STEPS] for i in range(len(seq3) - NUM_STEPS)])
DATES_TRAIN = D[:int((len(D)*.8))]
DATES_TEST = D[int((len(D)*.8))+1:]

# Initialize graph
tf.reset_default_graph()
lstm_graph = tf.Graph()

# Define the graph
with lstm_graph.as_default():
  inputs = tf.placeholder(tf.float32, [None, NUM_STEPS, VECTOR_SIZE])
  targets = tf.placeholder(tf.float32, [None, TARGET_SIZE])
  learning_rate = tf.placeholder(tf.float32, None)
  
  def _create_one_cell():
    return tf.contrib.rnn.LSTMCell(LSTM_SIZE)
    if KEEP_PROB < 1.0:
      return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
  
  cell = tf.contrib.rnn.MultiRNNCell([_create_one_cell() for _ in range(NUM_LAYERS)], state_is_tuple=True) if NUM_LAYERS > 1 else _create_one_cell()
  
  input_weights = tf.Variable(tf.truncated_normal([VECTOR_SIZE]))
  input_bias = tf.Variable(tf.constant(0.01, shape=[VECTOR_SIZE]))
  
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
    for batch_X, batch_Y in zip(list(chunks(X_TRAIN, BATCH_SIZE)), list(chunks(Y_TRAIN, BATCH_SIZE))):
      train_data_feed = {
          inputs: batch_X,
          targets: batch_Y,
          learning_rate: current_lr
      }
      train_loss, _ = sess.run([loss, minimize], train_data_feed)
      saver = tf.train.Saver()
      saver.save(sess, "./model.ckpt")
