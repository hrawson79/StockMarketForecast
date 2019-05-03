from pandas_datareader import data
from sklearn import linear_model
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler



def LoadStockDataForSymbol(symbol, start_date, end_date, source='yahoo'):
    # Use pandas_reader.data.DataReader to load the desired data
    panel_data = data.DataReader(symbol, source, start_date, end_date)

    # Get all of the weekdays within the range specified above
    all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

    # Reindex close using all_weekdays as new index
    panel_data = panel_data.reindex(all_weekdays)

    # Fill in missing dates
    panel_data = panel_data.fillna(method='ffill')

    return panel_data

class stock(object):

  def __init__(self, symbol, start_date, end_date, source='yahoo'):
    self.data_frame_ = LoadStockDataForSymbol(symbol, start_date, end_date, source)
    self.close_ = self.data_frame_['Close']
    self.open_ = self.data_frame_['Open']
    self.high_ = self.data_frame_['High']
    self.low_ = self.data_frame_['Low']
    self.volume_ = self.data_frame_['Volume']
    self.adj_close_ = self.data_frame_['Adj Close']
    self.dates_ = self.data_frame_.index.values
    self.close_mov_avg_20_ = self.close_.rolling(window=20).mean()
    self.close_mov_avg_100_ = self.close_.rolling(window=100).mean()

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
    self.test_inputs_ = self.inputs_[int((len(self.inputs_)*.8))+1:]

    self.train_targets_ = self.targets_[:int((len(self.targets_)*.8))]
    self.test_targets_ = self.targets_[int((len(self.targets_)*.8))+1:]

    self.train_dates_ = self.dates_[:int((len(self.dates_)*.8))]
    self.test_dates_ = self.dates_[int((len(self.dates_)*.8))+1:]

# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
symbol = 'TSLA'#^GSPC


# In[2]:


# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2016-02-18'
end_date = '2019-03-28'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
data = LoadStockDataForSymbol(symbol, start_date, end_date) # Here, data has 6 columns, High, Low, Open, Close and Volumn
dates = data.index.values
data = data[np.isfinite(data['High'])]
#print(data['Close'].values)
mean = np.mean(data['Close'].values)
std = np.std(data['Close'].values)
#print(dates)
#data_s = data['Close'].shift(-1) # Shifted data is the 'Close' column shifted downwards by 1 column, which means we use all 6 columns as feature and the next day's close value as label
#data = data.drop(data.index[len(data)-1]) # Make the dimension the same
#data_s = data_s[np.isfinite(data_s)] # Remove 'NaN' data


# In[3]:


# Standarize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)
#data_s = scaler.fit_transform(data_s)
#mean = np.mean(data_s, axis = 0)#normalize every row
#std = np.std(data_s, axis = 0)
#data_s = (data_s - mean) / std
data_s = data[:, 3]
data_s = np.roll(data_s, -1)
data_s = data_s.reshape(-1, 1)
data = np.delete(data, (0), axis=0)
data_s = np.delete(data_s, (0), axis=0)
#print(data)
#print(data_s)


# In[4]:


#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0, shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(data, data_s, test_size=0.2, random_state=0, shuffle=False)# Divide the data by 1:1
dates_train, dates_test = train_test_split(dates, test_size=0.2, random_state=0, shuffle=False)
#print(len(dates_train), len(dates_test), len(dates))


# In[5]:


# Train the model using the training sets
reg = LinearRegression().fit(x_train, y_train)
#clf_RF = RandomForestClassifier()
#reg = clf_RF.fit(x_train, y_train)

# Make predictions using the testing set
#y_pred_train = reg.predict(x_train)
y_pred_test = reg.predict(x_test)
y_test = y_test[:-1]
print(len(y_test))
y_pred_test = y_pred_test[:-1]
print(len(y_pred_test))


# In[6]:


fig, ax = plt.subplots(figsize=(16, 9))
plt.gca().tick_params(labelsize=16)
ax.plot(dates_test[1:(len(dates_test)-1)], y_test, label="test")
ax.plot(dates_test[1:(len(dates_test)-1)], y_pred_test, label="pred")
#ax.legend()
plt.rcParams.update({'font.size':20})
ax.legend(prop={'size':20})
ax.set_title('Test Data')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price', fontsize=18)
#plt.rcParams.update({'font.size':20})


# In[7]:


MSE = np.sum((y_test - y_pred_test) ** 2) / len(y_test)# calculate Mean Square Error
print(MSE)


# In[8]:


# prediction: y_pred_test
# target: y_test
print(mean)
print(std)
rate = 0
for i in range(1, len(y_test)-1):
    if (y_pred_test[i+1] > y_pred_test[i]): # Only add the rate if when predicting stock going up
        rate += (y_test[i+1] - y_test[i])/(y_test[i] + mean/std)
        #print(y_test[i] + mean/std)
avg = rate/len(y_test)
print(avg)






