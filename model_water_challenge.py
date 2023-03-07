from tensorflow.keras.metrics import RootMeanSquaredError as rmse
from tensorflow.keras.metrics import MeanAbsoluteError as mae

#MODELS
import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objs as go
import datetime
from time import time
import seaborn as sns
import pandas as pd
import numpy as np
import math 


# Plot imports
import plotly.graph_objs as go

## Offline mode
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from IPython.display import Image
from IPython.core.display import HTML 

# Visualization
import matplotlib.pyplot as plt
plt.style.use('bmh')
import plotly.express as px

## Offline mode
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

#time
from datetime import timedelta
import datetime

# PREPROCESSING 
def name_model_to_save(model_name,look_back,epochs):
    # name to save the model
    today= datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    name_model_to_save='{}_lookback-{}_epochs-{}_{}'.format(model_name,look_back,epochs,today)
    return name_model_to_save


def split_training_test_with_dates(df, start_train,end_train,end_test):

    # transform to datetime type
    train_end = datetime.datetime.strptime(end_train, '%Y-%m-%d')
    test_end = datetime.datetime.strptime(end_test, '%Y-%m-%d')
    
    # difference in days
    diff_in_days=train_end-test_end
    diff_in_days=diff_in_days.days

    # split training
    train_data = df.loc[start_train:end_train,:]

    #split test
    start_test= train_end + timedelta(days=1)
    test_start=start_test.strftime('%Y-%m-%d')
    test_data = df[test_start:end_test]
    
    return train_data,test_data,diff_in_days


def split_index_from_data(df_train,df_test):
    # drop na and index
    df_train = df_train.dropna()
    df_train_index = df_train.index

    df_train = df_train.reset_index(drop=True)

    df_test = df_test.dropna()
    df_test_index = df_test.index    

    df_test = df_test.reset_index(drop=True)
    return df_train, df_test_index, df_test, df_test_index


# MODEL
def create_LSTM_model(x_train,batch_size,opt):
        model = Sequential()
        # Expected input batch shape: (batch_size, timesteps, data_dim).
        '''
        batch_size depends in part on your specific problem, but mostly is given by the size of your dataset. 
        If you specify a batch size of x and your dataset contains N samples, during training your
         data will be split in N/x groups (batches) of size x each.

         batch size defines the number of samples that will be propagated through the network.

        timesteps (the size of your temporal dimension) or "frames" each sample sequence has, 

        data_dim (that is, the size of your data vector on each timestep).
        X = [[0.54, 0.3], [0.11, 0.2], [0.37, 0.81]] has a timestep of 3 and a data_dim of 2.


        For instance, let's say you have 1050 training samples and you want to set up a batch_size 
        equal to 100. The algorithm takes the first 100 samples (from 1st to 100th) 
        from the training dataset and trains the network. Next, it takes the second 
        100 samples (from 101st to 200th) and trains the network again. We can keep doing 
        this procedure until we have propagated all samples through of the network. 
        Problem might happen with the last set of samples. In our example, we've used 1050 
        which is not divisible by 100 without remainder. The simplest solution is just to 
        get the final 50 samples and train the network.

	Advantages of using a batch size < number of all samples:

		It requires less memory. Since you train the network using fewer samples, the 
		overall training procedure requires less memory. That's especially important if
		 you are not able to fit the whole dataset in your machine's memory.

		Typically networks train faster with mini-batches. That's because we update the
		weights after each propagation. In our example we've propagated 11 batches 
		(10 of them had 100 samples and 1 had 50 samples) and after each of them we've
		updated our network's parameters. If we used all samples during propagation we
		would make only 1 update for the network's parameter.

	Disadvantages of using a batch size < number of all samples:

		The smaller the batch the less accurate the estimate of the gradient will be
        '''
        model.add(LSTM(layers, batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer=opt,metrics=[rmse(), mae()])
        return model
    



def split_x_and_y_on_lookback_and_horizon(df_t,look_back=30,target_column,horizon=30): 
    x_data, y_data = [], []

    for i in range(len(df_t)-look_back):

        x = df_t[i:(i+look_back)].to_numpy()
        x_data.append(x)

        y=df_t[target_column].loc[i:(i+horizon-1)].to_numpy()
        y_data.append(y)

    x_t, y_t = np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32)
    print ('Test set: {}'.format(y_t.shape)) # number of 
    print ('Training set: {}'.format(x_t.shape))
    return x_t, y_t

def transform_dataset(df,scaler,y_value,x_values,target_column):
    dataset = np.zeros(shape=(len(y_value), x_values.shape[2]))
    dataset[:,df.columns.get_loc(target_column)] = y_value[:,0]
    transfomored_dataset = scaler.inverse_transform(dataset)[:,df.columns.get_loc(target_column)]

    return transfomored_dataset

def MEA(y_pred, y_true):
    y_pred=np.array(y_pred)
    y_true=np.array(y_true)

    output_errors = np.abs(y_pred - y_true)
    return output_errors

def RMSE(y_pred, y_true):
    y_pred=np.array(y_pred)
    y_true=np.array(y_true)
    output_errors = ((y_true - y_pred) ** 2)/len(y_true)
    output_errors=sqrt(output_errors)
    return output_errors        
def compute_errors(train_,test_):
    train_predict= train_['predicted_values']
    y_train_ =train_['real_values']

    test_predict= test_['predicted_values']
    y_test_ =test_['real_values']

    score_train_mae = mean_absolute_error(y_train_, train_predict)
    score_test_mae = mean_absolute_error(y_test_, test_predict)

    score_train_rsme = math.sqrt(mean_squared_error(y_train_, train_predict))
    score_test_rsme = math.sqrt(mean_squared_error(y_test_, test_predict))

    print(f'MAE Train: {score_train_mae} \nMAE Test: {score_test_mae}')
    print(f'RSME Train: {score_train_rsme} \nRSME Test: {score_test_rsme}')
    return score_train_mae, score_test_mae, score_train_rsme, score_test_rsme





def lstm_predict(df_norm,folder_name, model_name,target_column, look_back, layers, epochs, start_train,end_train,end_test, verbose=1):
    

	# built name to save the model      
	name_model_to_save=name_model_to_save(model_name,look_back,epochs)

	# split trainig-test set
	df_train,df_test,diff_in_days= split_training_test_with_dates(df, start_train,end_train,end_test)
	print('Training data\n    from {} to {}\n size: {}'.format(start_train,end_train, len(df_train)))
	print('Test data\n    from {} to {}\n size: {}'.format(start_train,end_train, len(df_test)))

	# save date index 
	df_train, df_test_index, df_test, df_test_index=split_index_from_data(df_train,df_test)

     
    # split independent from dependent variables 
    x_train, y_train = split_x_and_y_on_lookback_and_horizon(df_train,look_back,target_column)

    # split independent from dependent variables
    x_test, y_test = split_x_and_y_on_lookback_and_horizon(df_test,look_back,target_column)

        
    # create model with batch_size 
    model = create_LSTM_model(x_train,batch_size, opt)

    # FIT
    history=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    model.save('saved_model/{}/{}'.format(folder_name,model_name))

    # PREDICT
    train_predict = model.predict(x_train, batch_size=batch_size1)
    test_predict = model.predict(x_test, batch_size=batch_size)

    train_predict=transform_dataset(df,scaler,train_predict,x_train,target_column)
    test_predict=transform_dataset(df,scaler,test_predict,x_test,target_column)

     
    y_train_=transform_dataset(df,scaler,y_train,x_train,target_column)
    train_= {'predicted_values':train_predict, 'real_values':y_train_}
    train_ = pd.DataFrame(data=train_)
    df_train_index=df_train_index[look_back:]

    train_= train_.set_index(df_train_index)
    
    y_test_=transform_dataset(df,scaler,y_test,x_test,target_column)
    test_= {'predicted_values':test_predict, 'real_values':y_test_}
    test_ = pd.DataFrame(data=test_)
    df_test_index=df_test_index[look_back:]
    test_= test_.set_index(df_test_index)

    return train_, test_, model,df_train_index,df_test_index, history