import scripts.data.model_data as model_data
import scripts.data.stock_data as stock_data
import scripts.models.ltsm.lstm as lstm

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    # Stock symbols to be downloaded
    tickers = ['^GSPC']

    # Date range of data to be downloaded
    start_date = '2016-02-02'
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
    MAX_EPOCH = 100
    VECTOR_SIZE = 6

    stock_1 = stock_data.stock(tickers, start_date, end_date)
    data_1 = model_data.model_data(stock_1, INPUT_SIZE, NUM_STEPS)
    #print(stock_1.shape)

    # Initialize graph
    lstm_1 = lstm.LSTM()


    # Define the graph
    lstm_1.define()

    #print(data_1.test_targets_)
    # Train graph
    lstm_1.train(data_1.train_inputs_, data_1.train_targets_)

    # Test graph
    lstm_1.test(data_1.test_inputs_, data_1.test_targets_)

    print("End")

