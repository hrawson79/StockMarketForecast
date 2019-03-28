# This module creates a container for data sequence structure
# to be input into the model and targets to compare.
#
#

import numpy as np
import pandas as pd

class model_data(object):
  def __init__(self, stock_obj, input_size, num_steps):
    self.input_seq_ = [np.array(stock_obj.data_frame_[i * input_size: (i + 1) * input_size])
       for i in range(len(stock_obj.data_frame_) // input_size)]
    self.target_seq_ = [np.array(stock_obj.close_[i * input_size: (i + 1) * input_size])
       for i in range(len(stock_obj.close_) // input_size)]
    self.date_seq_ = [np.array(stock_obj.dates_[i * input_size: (i + 1) * input_size])
       for i in range(len(stock_obj.dates_) // input_size)]

    self.inputs_ = np.array([self.input_seq_[i: i + num_steps] / self.input_seq_[i-1]
       for i in range(len(self.input_seq_) - num_steps)])
    self.inputs_ = self.inputs_.reshape(len(self.inputs_), num_steps, len(stock_obj.data_frame_.columns))

    self.targets_ = np.array([self.target_seq_[i + num_steps] / self.target_seq_[i-1]
       for i in range(len(self.target_seq_) - num_steps)])
    self.targets_ = self.targets_.reshape(len(self.targets_), input_size)

    self.dates_ = np.array([self.date_seq_[i + num_steps]
       for i in range(len(self.date_seq_) - num_steps)])
    self.dates_ = self.dates_.reshape(len(self.dates_), input_size)

    self.train_inputs_ = self.inputs_[:int((len(self.inputs_)*.8))]
    self.test_inpts_ = self.inputs_[int((len(self.inputs_)*.8))+1:]

    self.train_targets_ = self.targets_[:int((len(self.targets_)*.8))]
    self.test_targets_ = self.targets_[int((len(self.targets_)*.8))+1:]

    self.train_dates_ = self.dates_[:int((len(self.dates_)*.8))]
    self.test_dates_ = self.dates_[int((len(self.dates_)*.8))+1:]
