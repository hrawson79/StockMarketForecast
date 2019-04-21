import scripts.data.model_data as model_data
import scripts.data.stock_data as stock_data
import scripts.models.ltsm.lstm as lstm

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob

if __name__ == "__main__":

    # Stock symbols to be downloaded
    # tickers = ['^GSPC']
    tickers = ['TSLA']
    # tickers = ['MSFT']
    ticker_name = tickers[0].replace('^', '')

    # Date range of data to be downloaded
    start_date = '2016-02-18'
    end_date = '2019-03-28'

    # # Clear the work directory
    # for filename in glob.glob("work/*"):
    #     os.remove(filename)

    stock_1 = stock_data.stock(tickers, start_date, end_date)
    data_1 = model_data.model_data(stock_1, lstm.INPUT_SIZE, lstm.NUM_STEPS)

    model_name = "%s_layer_%d_epoch_%d_%s_%s" % (ticker_name, lstm.NUM_LAYERS, lstm.MAX_EPOCH, start_date, end_date)

    fp = open("results/%s.txt" % (model_name), "w")

    print("Target STD: %f" % data_1.target_seq_std)
    print("Target STD: %f" % data_1.target_seq_std, file=fp)


    # Initialize graph
    lstm_1 = lstm.LSTM()
    

    # Define the graph
    lstm_1.define()

    
    # Train graph
    lstm_1.train(model_name, data_1.train_inputs_, data_1.train_targets_)

    # Test graph
    
    mse_train, mse_test = lstm_1.test(model_name, data_1.test_inputs_, data_1.test_targets_, data_1.train_inputs_, data_1.train_targets_)

    print("Train MSE: %f " % mse_train, file=fp)
    print("Test MSE: %f" % mse_test, file = fp)
    print("End")

