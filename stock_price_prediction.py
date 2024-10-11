#import packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.regularizers import L1L2
#import project modules
from stock_prediction_modules import *

###get data set and plot data set graphs
stock_market_dataset = plot_all_graphs('Google.csv')

###normalize data
normalized_array, normalized_df, normalizer = normalize_dataset(stock_market_dataset)
normalized_df.head()
n = normalized_df.shape[0]

###split data set into train, cross validation and test set

num_features = 6 
close_value_col_index = 3
#creating training set with time steps.

time_step=30    #1 month time step
train_per = 0.7
val_per = 0.2
test_per = 1-(train_per+val_per)  #0.1
    
X_train, y_train, X_val, y_val, X_test, y_test =  make_train_test_val_sets(normalized_array, num_features, close_value_col_index, time_step, train_per, val_per, test_per)

print(str(X_train.shape)+ " training dataset shape")
print(str(X_val.shape)+ "  val dataset shape")
print(str(X_test.shape)+ "  test dataset shape")

### Train model
#Implementing LSTM
model = keras.models.Sequential()         #initializing network

#input layer
hidden_layer_units = 10
model.add(keras.layers.LSTM(hidden_layer_units,kernel_regularizer=L1L2(0.0001), return_sequences=True, input_shape=(X_train.shape[1],num_features)))

#LSTM layer 2
hidden_layer_units = 10
model.add(keras.layers.LSTM(hidden_layer_units,kernel_regularizer=L1L2(0.0001)))

#Output layer
output_layer_units = 1     #just need the close value
model.add(keras.layers.Dense(output_layer_units))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 70, batch_size = 10)

#Get predictions on all sets
real_close_value = normalized_df.iloc[ time_step: , close_value_col_index:close_value_col_index+1].values    
print(real_close_value.shape)

predicted_close_value_train, predicted_close_value_val, predicted_close_value_test, combined_prediction = get_predictions(model, X_train, X_val, X_test)
print(combined_prediction.shape)

#inverse normalize predicted close values
denormalized_df = denormalize_data(normalized_df, normalizer, time_step, combined_prediction, close_value_col_index)
predicted_close_value = denormalized_df.iloc[ : ,close_value_col_index:close_value_col_index+1]
denormalized_df.head(5)

#Training set actual vs prediction
real_close_value_train = normalized_df.iloc[ time_step:int(n*train_per)+time_step , close_value_col_index:close_value_col_index+1].values
plot_versus_graph(real_close_value_train, 'blue', 'Close', predicted_close_value_train, 'red', 'Predicted Close')

#Cross Val set actual vs predicted
#int(n*0.7)+time_step
real_close_value_val = normalized_df.iloc[ int(n*0.7)+time_step:int(n*0.9)+time_step , close_value_col_index:close_value_col_index+1].values
plot_versus_graph(real_close_value_val, 'blue', 'Close', predicted_close_value_val, 'orange', 'Predicted Close')

#Test set actual vs predicted
#int(n*0.7)+time_step
real_close_value_test = normalized_df.iloc[ int(n*0.9)+time_step: , close_value_col_index:close_value_col_index+1].values
plot_versus_graph(real_close_value_test, 'blue', 'Close', predicted_close_value_test, 'green', 'Predicted Close')
