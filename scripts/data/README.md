# Data Model

The modules in this directory create containers to hold stock market data and input/target data for the LSTM model. These containers encapsulate the data in a way that eases accessibility when downloading stock data and shaping it to be input into the LSTM model. The following are the containers in each module and the attributes of each one.

# stock_data.py
## stock
* data_frame_
* close_
* open_
* high_
* low_
* volume_
* adj_close_
* dates_
* close_mov_avg_20_
* close_mov_avg_100_

# model_data.py
## model-data
* input_seq_
* target_seq_
* date_seq_
* inputs_
* targets_
* dates_
* train_inputs_
* test_inputs_
* train_targets_
* test_targets_
* train_dates_
* test_dates_
