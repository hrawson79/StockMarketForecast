import scripts.data.model_data as model_data
import scripts.data.stock_data as stock_data
import scripts.models.ltsm.lstm as lstm

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Stock symbols to be downloaded
    # tickers = ['^GSPC']
    tickers = ['TSLA']
    # tickers = ['MSFT']
    ticker_name = tickers[0].replace('^', '')

    # Date range of data to be downloaded
    start_date = '2016-02-18'
    end_date = '2019-03-28'

    INPUT_SIZE = 1
    TARGET_SIZE = 1
    NUM_STEPS = 30
    INIT_LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = 0.99
    BATCH_SIZE =256
    KEEP_PROB = 0.8
    LSTM_SIZE = 512
    NUM_LAYERS = 3
    INIT_EPOCH = 5
    MAX_EPOCH = 50
    VECTOR_SIZE = 6

    stock_1 = stock_data.stock(tickers, start_date, end_date)
    data_1 = model_data.model_data(stock_1, INPUT_SIZE, NUM_STEPS)

    fp = open("results/%s_layer_%d_epoch_%d_%s_%s.txt" % (ticker_name, NUM_LAYERS, MAX_EPOCH, start_date, end_date), "w")

    print("Target STD: %f" % data_1.target_seq_std)
    print("Target STD: %f" % data_1.target_seq_std, file=fp)


    # Initialize graph
    lstm_1 = lstm.LSTM()
    

    # Define the graph
    lstm_1.define()

    
    # Train graph
    lstm_1.train(data_1.train_inputs_, data_1.train_targets_)

    # Test graph
    mse_train, mse_test = lstm_1.test(data_1.test_inputs_, data_1.test_targets_, data_1.train_inputs_, data_1.train_targets_)

    plt.savefig("figures/%s_layer_%d_epoch_%d_%s_%s.png" % (ticker_name, NUM_LAYERS, MAX_EPOCH, start_date, end_date))

    print("Train MSE: %f " % mse_train, file=fp)
    print("Test MSE: %f" % mse_test, file = fp)
    print("End")

