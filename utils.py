# Scraping
from urllib.request import urlopen, Request
# from bs4 import BeautifulSoup as soup
import re
import time
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import datetime
from datetime import timedelta

import math 
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px


# Plot imports
import plotly.graph_objs as go

## Offline mode
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')





# plot map
from sklearn import preprocessing
import folium
from geopy.geocoders import Nominatim

# acf and pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

#models
import math
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.metrics import RootMeanSquaredError as rmse
from tensorflow.keras.metrics import MeanAbsoluteError as mae

class WaterSource:
    def __init__(self, water_source_name):
        self.water_source_name = water_source_name
        self.water_source_title = self.water_source_name.replace('_', ' ')
        self.water_source_csv_name = "./data/original/{}.csv".format(water_source_name)      
        self.water_source_csv= pd.read_csv(self.water_source_csv_name, index_col=0)
#         self.dependent_features =['Depth_to_Groundwater_Podere_Casetta', 'Depth_to_Groundwater_Pozzo_1', 'Depth_to_Groundwater_Pozzo_3', 'Depth_to_Groundwater_Pozzo_4']
        
        self.set_dependent()
        
    def Aquifer():
        return ['Aquifer_Auser_Depth_to_Groundwater_LT2','Aquifer_Auser_Depth_to_Groundwater_SAL', 'Aquifer_Auser_Depth_to_Groundwater_PAG', 'Aquifer_Auser_Depth_to_Groundwater_CoS', 'Aquifer_Auser_Depth_to_Groundwater_DIEC', 'Aquifer_Luco_Depth_to_Groundwater_Podere_Casetta', 'Aquifer_Luco_Depth_to_Groundwater_Pozzo_1', 'Aquifer_Luco_Depth_to_Groundwater_Pozzo_3', 'Aquifer_Luco_Depth_to_Groundwater_Pozzo_4', 'Aquifer_Petrignano_Depth_to_Groundwater_P24', 'Aquifer_Petrignano_Depth_to_Groundwater_P25', 'Aquifer_Doganella_Depth_to_Groundwater_Pozzo_1', 'Aquifer_Doganella_Depth_to_Groundwater_Pozzo_2', 'Aquifer_Doganella_Depth_to_Groundwater_Pozzo_3', 'Aquifer_Doganella_Depth_to_Groundwater_Pozzo_4', 'Aquifer_Doganella_Depth_to_Groundwater_Pozzo_5', 'Aquifer_Doganella_Depth_to_Groundwater_Pozzo_6', 'Aquifer_Doganella_Depth_to_Groundwater_Pozzo_7', 'Aquifer_Doganella_Depth_to_Groundwater_Pozzo_8', 'Aquifer_Doganella_Depth_to_Groundwater_Pozzo_9']

    def Lake():
        return ['Lake_Level','Flow_Rate']
        
    def River():
        return ['Hydrometry_Nave_di_Rosano']
        
    
    def set_dependent(self):
        if self.water_source_name=='merged_Aquifer':
            self.dependent_features = WaterSource.Aquifer()
        elif self.water_source_name=='Lake_Bilancino':
            self.dependent_features = WaterSource.Lake()
        elif self.water_source_name=='River_Arno':
            self.dependent_features = WaterSource.River()
    def get_dependent(self):
        return self.dependent_features

        


def firstNonNan(column_items, column_name):
    for index  in column_items.index:
        item = column_items.loc[index,column_name]
        if math.isnan(item) == False:
              return item ,  index
            
def get_firstNaN_df(df):
    for column_name in df:
        column_items = df[[column_name]]
        item, index = firstNonNan(column_items,column_name )
        print( '{} started on {} \n'.format(column_name, 
                                                datetime.datetime.strptime(index, '%Y-%m-%d').strftime('%B %d, %Y')) ) 

def built_time_feature(df):

    df['year'] = pd.DatetimeIndex(df.index).year
    df['month'] = pd.DatetimeIndex(df.index).month
    df['day_of_year'] = pd.DatetimeIndex(df.index).dayofyear
    df['week_of_year'] = pd.DatetimeIndex(df.index).weekofyear
    df['quarter'] = pd.DatetimeIndex(df.index).quarter
    df['season'] = df.month%12 // 3 + 1
    

def extract_feature_amount(df, num_target_column, range_time='week', how_many=1, type_result='sum'):
    # convert colum name to number 
    if num_target_column is not int:
        num_target_column = df.columns.get_loc(num_target_column)
    elif num_target_column is  int:
        pass
    else:
        print("Warining: {} is not in dataset".format(num_target_column))
    len_df = len(df)
    
    # choose the range of time
    if range_time=='week':
        start= 7*how_many

    if range_time=='month':
        start= 30*how_many       
           
    if range_time=='year':
        start= 365*how_many
        
    r_end= start+1
    
    total_days = range(0,r_end)
    end=len_df-start
    
    # to solve the problem of infer on data range_time     
    total_previous=[np.nan] * start 

    # range from start (previous ones are np.nan and end of the df)
    for each in range(start,len_df):
        # temp list for computing the amount in a specific range of time (ex. week)
        total =[]
        
        #range of time for each value in the total_days
        for i in total_days:
#             print(total)
            try:
                # for the current data point we subtract i values (we want the past week)
                total.append(df.iloc[each-i,num_target_column]) 
            except:
                total.append(np.nan)
     
        if type_result == 'sum':    
            total_previous.append(sum(total))
            
        if type_result == 'mean':    
            total_previous.append(np.mean(total))
            
    # to solve the problem of infer on data range_time
#     print(len_df-len(total_previous))
    return total_previous 


def drop_and_reorder_columns(df, first_cols=[], last_cols=[], drop_cols=[]):
    columns = df.columns.tolist()
    columns = list(set(columns) - set(first_cols))
    columns = list(set(columns) - set(drop_cols))
    columns = list(set(columns) - set(last_cols))
    new_order = first_cols + columns + last_cols
    df = df[new_order]
    return df




def extract_total_features(df, str_target, type_result='mean'):
    if type(str_target)==str:
        target_columns = df.columns[df.columns.str.contains(str_target)]
    elif type(str_target)==list:
        target_columns = str_target
    if type_result == 'mean':
        return df[target_columns].mean(axis=1)
    elif type_result == 'sum':
        return df[target_columns].sum(axis=1)


### Visualization Functions

def NaN_analysis(df, title):
    palette= sns.color_palette("pastel")

    temp = df.isnull().sum()

    plt.figure(figsize=(15,3))
    g = sns.barplot(temp.index, temp.values)
    plt.xticks(rotation=90)
    plt.ylim(0,(temp.values.max()+1000))
    plt.title(title)
    
    for p in g.patches:
        g.annotate('{:.0f}\n{:.2f}'.format(p.get_height(), (p.get_height()/df.shape[0]) ), 
                   (p.get_x()+0.4, p.get_height()+10),
                    ha='center', va='bottom',
                    color= 'black')
    
    plt.show()

def plot_ACF_PACF(water_source,target_columns,lag_ACF, lag_PACF):
    for dependent_feature in target_columns:
        fig = plt.figure(figsize=(12,8))

        not_nan_water_source,starting_date,ending_date=get_not_nan_water_source(water_source,dependent_feature)

        # get autocorrelation & partial autocorrelation
        ax1 = fig.add_subplot(211)
        fig = plot_acf(not_nan_water_source[dependent_feature],lags=lag_ACF,ax=ax1, title= 'CORRELOGRAM for {}:\n Autocorrelation:'.format(dependent_feature))
        ax2 = fig.add_subplot(212)
        fig = plot_pacf(not_nan_water_source[dependent_feature],lags=lag_PACF,ax=ax2, title= 'Partial Autocorrelation:'.format(dependent_feature))




def get_not_nan_water_source(water_source,dependent_feature, feat_only=False):  # need to subset the dataframe in order to not have NaN.
    column_series = water_source.loc[:,dependent_feature]
    starting_date=column_series.first_valid_index()
    ending_date=column_series.last_valid_index()
    if (feat_only==False):
        not_nan_water_source=water_source[starting_date:ending_date]
    else:
        not_nan_water_source=pd.DataFrame(water_source.loc[starting_date:ending_date,dependent_feature])
  
    return not_nan_water_source,starting_date,ending_date


def boxplot_seasonality(df, str_target, target_name_title_measure=False):
    
        dict_features_measures ={
        'Rainfall':'(mm)',
        'Depth_to_Groundwater': '(m from ground)',
        'Temperature':'(°C)',
        'Volume': '(m cubic)',
        'Hydrometry':'(m)',
        'Flow_Rate':'(l/s)',
        'Lake_Level':'(m)'}
        
    
        # extract columns which contains str_target
        target_columns = df.columns[df.columns.str.contains(str_target)]
        
        for feature in dict_features_measures.keys():
            if str_target in feature:
                
                target_name_title_measure = dict_features_measures[feature]
                target_name_title = feature.replace('_', ' ')
            elif  feature in str_target:
                target_name_title_measure = dict_features_measures[feature]
                target_name_title = feature.replace('_', ' ')
            else:
                continue

        if target_name_title_measure == False:
            print('Warning: need to specify a target_name_title_measure')
            

        # create a list of natural language tag refering to each target column 
        target_columns_name_list = [ x.replace(feature,'').replace('_', ' ')   if (x != feature) else target_columns[0].replace('_', ' ') for x in target_columns ]

       
        
        df_series = df.loc[:, (target_columns)].copy()
        df['year_month'] = pd.to_datetime(df.index)
        df['Months'] = df['year_month'].dt.month
 

        for str_target in target_columns:

            temp_water=px.box(df, x='Months', y=str_target, title='Month-wise Box Plot\n(The Seasonality): {}'
                              .format(target_name_title.replace('_', ' ')),
                                  )

            # Create a layout with interactive elements and two yaxes
            layout = go.Layout(height=800, width=1000, 
                               font=dict(size=10),
                               )


            fig = go.Figure(data=temp_water, layout=layout)
            iplot(fig)
        df=drop_and_reorder_columns(df, first_cols=[], last_cols=[], drop_cols=['year_month', 'Months',])


def boxplot_trend(df, str_target, target_name_title_measure=False):
    
        dict_features_measures ={
        'Rainfall':'(mm)',
        'Depth_to_Groundwater': '(m from ground)',
        'Temperature':'(°C)',
        'Volume': '(m cubic)',
        'Hydrometry':'(m)',
        'Flow_Rate':'(l/s)',
        'Lake_Level':'(m)'}
        
    
        # extract columns which contains str_target
        target_columns = df.columns[df.columns.str.contains(str_target)]
        
        for feature in dict_features_measures.keys():
            if str_target in feature:
                
                target_name_title_measure = dict_features_measures[feature]
                target_name_title = feature.replace('_', ' ')
            elif  feature in str_target:
                target_name_title_measure = dict_features_measures[feature]
                target_name_title = feature.replace('_', ' ')
            else:
                continue
        

        if target_name_title_measure == False:
            print('Warning: need to specify a target_name_title_measure')
            

        # create a list of natural language tag refering to each target column 
        target_columns_name_list = [ x.replace(feature,'').replace('_', ' ')   if (x != feature) else target_columns[0].replace('_', ' ') for x in target_columns ]

       
        
        df_series = df.loc[:, (target_columns)].copy()
        df['year_month'] = pd.to_datetime(df.index)
        df['Years'] = df['year_month'].dt.year
 

        for str_target in target_columns:

            temp_water=px.box(df, x='Years', y=str_target, title='Year-wise Box Plot\n(The Trend): {}'
                              .format(target_name_title.replace('_', ' ')),
                                  )

            # Create a layout with interactive elements and two yaxes
            layout = go.Layout(height=800, width=1000, 
                               font=dict(size=10),
                               )


            fig = go.Figure(data=temp_water, layout=layout)
            iplot(fig)
        df=drop_and_reorder_columns(df, first_cols=[], last_cols=[], drop_cols=['year_month', 'Years'])


def plot_seasonality(df,str_target):
    dict_features_measures ={
        'Rainfall':'(mm)',
        'Depth_to_Groundwater': '(m from ground)',
        'Temperature':'(°C)',
        'Volume': '(m cubic)',
        'Hydrometry':'(m)',
        'Flow_Rate':'(l/s)',
        'Lake_Level':'(m)'}
    for feature in dict_features_measures.keys():
            if feature in str_target:
                
                target_name_title_measure = dict_features_measures[feature]
                target_name_title = str_target.replace('_', ' ')

            else:
                continue
    df['year_month'] = pd.to_datetime(df.index)
    df['Years'] = df['year_month'].dt.year
    df['Months'] = df['year_month'].dt.month




    # palette = sns.color_palette("ch:2.5,-.2,dark=.3", 10)
    target_columns = df.columns[df.columns.str.contains(str_target)]
    for str_target in target_columns:
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.lineplot(df['Months'], df[str_target], hue=df['Years'])#, palette=palette)
        ax.set_title('Seasonal plot for {}'.format(str_target.replace('_',' ')), fontsize = 20, loc='center', fontdict=dict(weight='bold'))
        ax.set_xlabel('Months', fontsize = 16, fontdict=dict(weight='bold'))
        ax.set_ylabel('{} {}'.format(str_target.replace('_',' '), target_name_title_measure), fontsize = 16, fontdict=dict(weight='bold'))
        fig.show()
    df=drop_and_reorder_columns(df, first_cols=[], last_cols=[], drop_cols=['year_month', 'Months','Years'])


def plot_variables_against_time(df, str_target, target_name_title_measure=False):
    
        dict_features_measures ={
        'Rainfall':'(mm)',
        'Depth_to_Groundwater': '(m from ground)',
        'Temperature':'(°C)',
        'Volume': '(m cubic)',
        'Hydrometry':'(m)',
        'Flow_Rate':'(l/s)',
        'Lake_Level':'(m)'}
        
    
        # extract columns which contains str_target
        target_columns = df.columns[df.columns.str.contains(str_target)]
        
        for feature in dict_features_measures.keys():
            if str_target in feature:
                
                target_name_title_measure = dict_features_measures[feature]
                target_name_title = feature.replace('_', ' ')
            elif  feature in str_target:
                target_name_title_measure = dict_features_measures[feature]
                target_name_title = feature.replace('_', ' ')
            else:
                continue
        

        if target_name_title_measure == False:
            print('Warning: need to specify a target_name_title_measure')
            

        # create a list of natural language tag refering to each target column 
        target_columns_name_list = [ x.replace(feature,'').replace('_', ' ')   if (x != feature) else target_columns[0].replace('_', ' ') for x in target_columns ]

       
        
        df_series = df.loc[:, (target_columns)].copy()
        temp_water_list=[]

        for feature_number in range (df_series.shape[1]):

            # Create the same data object
            temp_water = go.Scatter(
                            x=df_series.index,
                            y=df_series.values[:,feature_number],
                            line=go.scatter.Line( width = 1.9),
                            opacity=0.8,
                            name='{}'.format(target_columns_name_list[feature_number]),
                            text=['{}: {} {}'.format(
                                                target_columns_name_list[feature_number] ,x, 
                                                target_name_title_measure) for x in df_series.values[:,feature_number]])

            temp_water_list.append(temp_water)

        # Create a layout with interactive elements and two yaxes
        layout = go.Layout(height=800, width=1000, 
                           font=dict(size=10),
                           title='Interactive Plot: {} versus time'.format(target_name_title),
                           xaxis=dict(title='Date',
                                        # Range selector with buttons
                                         rangeselector=dict(
                                             # Buttons for selecting time scale
                                             buttons=list([
                                                 
                                                 # Entire scale
                                                 dict(step='all'),
                                                 
                                                 # 1 year
                                                 dict(count=1,
                                                      label='1 year',
                                                      step='year',
                                                      stepmode='backward'),
                                                 
                                                 # 6 month
                                                 dict(count=6,
                                                      label='6 months',
                                                      step='month',
                                                      stepmode='backward'),
                                                 
                                                 # 1 month
                                                 dict(count=1,
                                                      label='1 month',
                                                      step='month',
                                                      stepmode='backward'),
                                                 # 1 week
                                                 dict(count=7,
                                                      label='1 week',
                                                      step='day',
                                                      stepmode='todate'),
                                                 # 1 day
                                                 dict(count=1,
                                                      label='1 day',
                                                      step='day',
                                                      stepmode='todate'),

                                             ])
                                         ),
                                         # Sliding for selecting time window
                                         rangeslider=dict(visible=True),
                                         # Type of xaxis
                                         type='date'),
                           yaxis=dict(title='{} {}'.format(target_name_title, 
                                                           target_name_title_measure), color='black'),


                           )


        fig = go.Figure(data=temp_water_list, layout=layout)
        iplot(fig)



### Scraping Functions



def keep_digits_and_dots(text):
    text =str(text)
    try:
        p = re.compile(r'<tr class="light"><td align="left" height="25px">Temperatura minima</td><td align="left">(\-?\d+) °C</td> </tr>')#<td height="25px" align="left">Temperatura media</td><td align="left">13 °C</td>
        text = p.findall(text)[0]
        
    except:
        pass
    return text

def request_urls(list_urls,list_dates):
    temperatures=pd.DataFrame(columns=['Date', 'Temperature'])
    error = []
    for i in range(len(list_urls)):
        url=list_urls[i]
        date=list_dates[i]
        
        #opening up connection, grabbing the page
        #my_url = 'https://www.ilmeteo.it/portale/archivio-meteo/Monte+Porzio+Catone/2014/Dicembre/1'
        try:
            r = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urlopen(r).read()

            #html parsing
            page_soup = soup(response, "html.parser")
            temperature = keep_digits_and_dots(page_soup.find('tr', {'class' : 'light'}))
            
            # add data
            temperatures.loc[i,'Temperature']= temperature
            temperatures.loc[i,'Date']= date

        except:
            error.append(date)
            print('WarningError: check'.format(date))
            pass
        

        # time to sleep
        time.sleep(0.2)          
    return temperatures, error


def convert_months_string_to_number(w_month):
    months_string_to_number = {
            '1':'Gennaio',
            '2': 'Febbraio',
            '3':'Marzo',
            '4':'Aprile',
            '5':'Maggio',
            '6':'Giugno',
            '7':'Luglio',
            '8':'Agosto',
            '9':'Settembre',
            '10':'Ottobre',
            '11':'Novembre',
            '12':'Dicembre'}

    for month_string in months_string_to_number.keys():
        if w_month == month_string:
            w_month = months_string_to_number.get(month_string)
            
        else:
            continue
        return w_month
    
def convert_months_number_to_string(w_month):
    months_number_to_string = {
            'Gennaio':'01',
            'Febbraio':'02',
            'Marzo':'03',
            'Aprile':'04',
            'Maggio':'05',
            'Giugno':'06',
            'Luglio':'07',
            'Agosto':'08',
            'Settembre':'09',
            'Ottobre':'10',
            'Novembre':'11',
            'Dicembre':'12'}

    for month_string in months_number_to_string.keys():
        if w_month == month_string:
            w_month = months_number_to_string.get(month_string)
            
        else:
            continue
        return w_month


def extract_url_from_date(null_water_source, url_source):
    list_urls=[]
    list_dates=[]
    
    for each_row in range(len(null_water_source)):
        w_year= null_water_source['Date'].dt.year[each_row]
        w_month= null_water_source['Date'].dt.month[each_row]
        w_month=convert_months_string_to_number(str(w_month))

        w_day= null_water_source['Date'].dt.day[each_row]
        w_day='%02d' % w_day # to add leading zero
        temp_url = '{}/{}/{}/{}'.format(url_source,w_year,w_month,w_day)
        list_urls.append(temp_url)
        date= '{}-{}-{}'.format(w_year,convert_months_number_to_string(w_month),w_day)
        list_dates.append(date)

    return list_urls,list_dates


def get_not_nan_water_source(water_source,dependent_feature, feat_only=False):  # need to subset the dataframe in order to not have NaN.
    column_series = water_source.loc[:,dependent_feature]
    starting_date=column_series.first_valid_index()
    ending_date=column_series.last_valid_index()
    if (feat_only==False):
        not_nan_water_source=water_source[starting_date:ending_date]
    else:
        not_nan_water_source=pd.DataFrame(water_source.loc[starting_date:ending_date,dependent_feature])
  
    return not_nan_water_source,starting_date,ending_date

def interpolate_feature(df, feature, inplace=True):
    old_feature= 'old version {}'.format(feature)
    column_series = df.loc[:,feature]
    starting_date=column_series.first_valid_index()
    ending_date=column_series.last_valid_index()

    df.loc[starting_date:ending_date, old_feature]=df.loc[starting_date:ending_date, feature].values
    df.loc[starting_date:ending_date, feature]=df.loc[starting_date:ending_date, old_feature].interpolate()
    plot_variables_against_time(df, feature)

    df.drop(old_feature, axis=1, inplace=inplace)
    return df

### Model Functions
def compare_dataframes(df_1, df_2, feat_1, feat_2, new_feat, starting_date, ending_date):
    # extract indeces from first df the period starting_date:ending_date
    indeces_df_1 =df_1.loc[starting_date:ending_date, feat_1].index
    indeces_df_2 =df_2.loc[starting_date:ending_date, feat_2].index

    # for each index
    for ind in indeces_df_1:
        try:
            # add the value of the df_2 to df_1
            if ind in indeces_df_2:
                df_1.loc[ind, new_feat]=df_2.loc[ind, feat_2]
        except:
            print('WarningError: {}'.format(ind))


def swift_feature(df_1,  feat_1, feat_2, starting_date, ending_date):
    # extract indeces from first df the period starting_date:ending_date
    df_1[feat_1]=df_1[feat_1].interpolate()
    temp_feat_1 =df_1.loc[starting_date:ending_date, feat_1].values
    
    df_1[feat_2]=df_1[feat_2].interpolate()
    temp_feat_2 =df_1.loc[starting_date:ending_date, feat_2].values    
    
    temp_water_source=temp_feat_1-temp_feat_2
    mean=temp_water_source.mean()
    sw_feat_2='Swifted_{}'.format(feat_2)
    df_1.loc[:,sw_feat_2]=df_1.loc[:, feat_2].values + mean





def shape_df(df_for_training_scaled,df_for_training, n_future = 14, n_past = 14):
    #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
    #In this example, the n_features is 2. We will make timesteps = 3. 
    #With this, the resultant n_samples is 5 (as the input data has 9 rows).
    trainX = []
    trainY = []

        # n_future = 1  Number of days we want to predict into the future
        # n_past = 14 Number of past days we want to use to predict the future

    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
    trainX, trainY = np.array(trainX), np.array(trainY)
    return trainX, trainY




def download_sir_data(df,dict_id_localities):
    downloaded_features0_24 = pd.DataFrame(columns=['date'])
    for ids in dict_id_localities.keys():
        try: 
            rainfall_context = urlopen(f'https://www.sir.toscana.it/archivio/download.php?IDST=pluvio0_24&IDS={ids}').read().decode('utf-8')
            to_add = pd.DataFrame([x.split(';') for x in rainfall_context[rainfall_context.find("gg/mm/aaaa") -1:].replace('@', '').replace(',','.').split('\r\n')])
            to_add = to_add.iloc[1:, :-1]
            to_add.columns = ['date', 'Rainfall_{}'.format(dict_id_localities.get(ids))]

            downloaded_features0_24 = downloaded_features0_24.merge(to_add, on='date', how='outer')
        except :
            print ('RequestError')
        try: 
            temperature_context = urlopen(f'https://www.sir.toscana.it/archivio/download.php?IDST=termo_csv&&IDS={ids}').read().decode('utf-8')
            to_add = pd.DataFrame([x.split(';') for x in temperature_context[temperature_context.find("gg/mm/aaaa") -1:].replace('@', '').replace(',','.').split('\r\n')])
            to_add = to_add.iloc[1:, :-1]
            to_add.columns = ['date', 'Temperature_{}'.format(dict_id_localities.get(ids))]

            downloaded_features0_24 = downloaded_features0_24.merge(to_add, on='date', how='outer')
        except :

            print ('RequestError')
    downloaded_features0_24['date']=downloaded_features0_24['date'].apply(lambda x: x.replace("/", "-"))
    df['date']=pd.to_datetime(df.index)
    downloaded_features0_24['date'] = pd.to_datetime(downloaded_features0_24['date'])#, format = '%d-%m-%Y', errors= 'coerce').strftime('%Y-%m-%d')
    merged_df = df.merge(downloaded_features0_24, on='date', how='outer')
    merged_df=merged_df.set_index('date')
    return merged_df

def merge_swifted_df(df, inplace= True):
    for feat_1 in df:
        if 'x' in feat_1:
            
            column_series = df.loc[:,feat_1]
            starting_date=column_series.first_valid_index()
            column_series = pd.Series(df.loc[:,feat_1])
            ending_date =  column_series.index[column_series.nonzero()[0][-1]]
            norm_feat= feat_1.replace('_x','')
            sw_feat = 'Swifted_{}'.format(feat_1.replace('_x','_y'))
            feat_2 = feat_1.replace('_x','_y')
            
           

            swift_feature(df, feat_1=feat_2,feat_2=feat_2,starting_date= starting_date, ending_date=ending_date)#,meanzero=True)
            df.loc[ending_date:,feat_1]=df.loc[ending_date:,sw_feat].values
            df.loc[:,norm_feat] = df.loc[:,feat_1]
            if inplace== True:
                df = drop_and_reorder_columns(df, first_cols=[],  drop_cols=[feat_2,sw_feat,feat_1])

def plot_locatons(df):
    locations = {}

    locations['Settefrati'] = {'lat' : 41.669624, 'lon' : 13.850011 }
    locations['Velletri'] = {'lat' : 41.6867015, 'lon' : 12.7770433 }
    locations['Petrignano'] = {'lat' : 43.1029282, 'lon' : 12.5237369 }
    locations['Piaggione'] = {'lat' : 43.936794, 'lon' : 10.5040929 }
    locations['S_Fiora'] = {'lat' : 42.854, 'lon' : 11.556 }
    locations['Abbadia_S_Salvatore'] = {'lat' : 42.8809724, 'lon' : 11.6724203 }
    locations['Vetta_Amiata'] = {'lat' : 42.8908958, 'lon' : 11.6264863 }
    locations['Castel_del_Piano'] = {'lat' : 42.8932352, 'lon' : 11.5383804 }
    locations['Terni'] = {'lat' : 42.6537515, 'lon' : 12.43981163 }
    locations['Bastia_Umbra'] = {'lat' : 43.0677554, 'lon' : 12.5495816  }
    locations['S_Savino'] = {'lat' : 43.339, 'lon' : 11.742 }
    locations['Monteroni_Arbia_Biena'] = {'lat' : 43.228279, 'lon' : 11.4021433 }
    locations['Monticiano_la_Pineta'] = {'lat' : 43.1335066 , 'lon' : 11.2408464 }
    locations['Montalcinello'] = {'lat' : 43.1978783, 'lon' : 11.0787906 }
    locations['Sovicille'] = {'lat' : 43.2806018, 'lon' : 11.2281756 }
    locations['Simignano'] = {'lat' : 43.2921965, 'lon' : 11.1680079 }
    locations['Mensano'] = {'lat' : 43.3009594 , 'lon' : 11.0548528 }
    locations['Siena_Poggio_al_Vento'] = {'lat' : 43.1399762, 'lon' : 11.3832092 }
    locations['Scorgiano'] = {'lat' : 43.3521445 , 'lon' : 11.15867 }
    locations['Ponte_Orgia'] = {'lat' : 43.2074581 , 'lon' : 11.2504416 }
    locations['Pentolina'] = {'lat' : 43.1968029, 'lon' : 11.1754672 }
    locations['Montevarchi'] = {'lat' : 43.5234999, 'lon' : 11.5675911 }
    locations['Incisa'] = {'lat' : 43.6558723, 'lon' : 11.4526838 }
    locations['Camaldoli'] = {'lat' : 43.7943293, 'lon' : 11.8199481 }
    locations['Bibbiena'] = {'lat' : 43.6955475, 'lon' : 11.817341 }
    locations['Stia'] = {'lat' : 43.801537, 'lon' : 11.7067347 }
    locations['Laterina'] = {'lat' : 43.5081823, 'lon' : 11.7102588 }
    locations['Monteporzio'] = {'lat' : 41.817251, 'lon' : 12.7050839 }
    locations['Pontetetto'] = {'lat' : 43.8226294, 'lon' : 10.4940843 }
    locations['Ponte_a_Moriano'] = {'lat' : 43.9083609 , 'lon' : 10.5342488 }
    locations['Calavorno'] = {'lat' : 44.0217216, 'lon' : 10.5297323 }
    locations['Borgo_a_Mozzano'] = {'lat' : 43.978948, 'lon' : 10.545703  }
    locations['Gallicano'] = {'lat' : 44.0606512, 'lon' : 10.435668  }
    locations['Tereglio_Coreglia_Antelminelli'] = {'lat' : 44.0550548 , 'lon' : 10.5623594 }
    locations['Lucca_Orto_Botanico'] = {'lat' : 43.84149865, 'lon' : 10.51169066 }
    locations['Orentano'] = {'lat' : 43.7796506, 'lon' : 10.6583892 }
    locations['Fabbriche_di_Vallico'] = {'lat' : 43.997647, 'lon' : 10.4279  }
    locations['Monte_Serra'] = {'lat' : 43.750833, 'lon' : 10.555278 }
    locations['Mangona'] = {'lat' : 44.0496863, 'lon' : 11.1958797 }
    locations['Le_Croci'] = {'lat' : 44.0360503, 'lon' : 11.2675661 }
    locations['Cavallina'] = {'lat' : 43.9833515, 'lon' : 11.2323312 }
    locations['S_Agata'] = {'lat' : 43.9438247, 'lon' : 11.3089835 }
    locations['Firenze'] = {'lat' : 43.7698712, 'lon' : 11.2555757 }
    locations['S_Piero'] = {'lat' : 43.9637372, 'lon' : 11.3182991 }
    locations['Vernio'] = {'lat' : 44.0440508 , 'lon' : 11.1498804  }
    locations['Consuma'] = {'lat' : 43.784, 'lon' : 11.585 }
    locations['Croce_Arcana']  = {'lat' : 44.1323056, 'lon' : 10.7689152 }
    locations['Laghetto_Verde']  = {'lat' :   42.883, 'lon' : 11.662  }

    locations_df = pd.DataFrame(columns=['city', 'lat', 'lon'] )

    def get_location_coordinates(df, column_type, cluster, target_df):
        for location in df.columns[df.columns.str.startswith(column_type)]:
            location = location.split(column_type)[1]

            loc_dict = {}
            loc_dict['city'] = location
            loc_dict['cluster'] = cluster
            loc_dict['lat'] = locations[location]['lat']
            loc_dict['lon'] = locations[location]['lon']

            target_df = target_df.append(loc_dict, ignore_index=True)

        return target_df

    locations_df = get_location_coordinates(df, 'Temperature_', 'aquifer_auser_df', locations_df)
    locations_df = get_location_coordinates(df, 'Rainfall_', 'aquifer_auser_df', locations_df)



    # Drop duplicates
    locations_df = locations_df.sort_values(by='city').drop_duplicates().reset_index(drop=True)

    # Label Encode cluster feature for visualization puposes
    le = preprocessing.LabelEncoder()
    le.fit(locations_df.cluster)
    locations_df['cluster_enc'] = le.transform(locations_df.cluster)

    m = folium.Map(location=[42.6, 12.4], tiles='cartodbpositron',zoom_start=7)

    colors = ['purple','lightred','green', 'lightblue', 'red', 'blue', 'darkblue','lightgreen', 'orange',  'darkgreen', 'beige',  'pink', 'darkred', 'darkpurple', 'cadetblue',]

    geolocator = Nominatim(user_agent='myapplication')
    for i in locations_df.index:
        folium.Marker([locations_df.iloc[i].lat, 
                      locations_df.iloc[i].lon],
                      popup=locations_df.iloc[i].city, 
                      icon=folium.Icon(color=colors[locations_df.iloc[i].cluster_enc])).add_to(m)

    return m


#Ho: It is non stationary
#H1: It is stationary
from statsmodels.tsa.stattools import adfuller
def adfuller_test(df, column_name):
    result=adfuller(df[column_name])
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    print('""""""""""" Stationary test for {} variable """""""""""'.format(column_name))
    for value,label in zip(result,labels):

        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("Stationary: Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary\n\n\n\n")
    else:
        print("Not Stationary:: Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary\n\n\n\n")



def subtraction_lag_feature(water_source,features,lags):
    lag_water_source=pd.DataFrame()
    for dependent_feature in features:
        for lag in lags:
            dependent_feature_before = 'Diff_{}_lag_{}'.format(lag,dependent_feature)
            not_nan_water_source,starting_date,ending_date=get_not_nan_water_source(water_source,dependent_feature, feat_only=True)
            
    #         not_nan_water_source=not_nan_water_source.loc[:,dependent_feature]
            not_nan_water_source[dependent_feature_before] = np.concatenate( [not_nan_water_source[dependent_feature] - not_nan_water_source[dependent_feature].shift(lag)        ])
            shifted_feature= not_nan_water_source[dependent_feature_before]
            lag_water_source = pd.concat([lag_water_source,shifted_feature ], axis=1)
     
    return lag_water_source


# MODEL
def create_LSTM_model(x_train,layers):
        model = Sequential()
        model.add(LSTM(layers, batch_input_shape=(1, x_train.shape[1], x_train.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics=[rmse(), mae()])
        return model
    
     





def plot_Scatter(x,y,error,error_values,model):
        # Create the same data object
        plot_data = go.Scatter(
        x=x,             
        y=y,
        line=go.scatter.Line( width = 1.9),
        opacity=0.8,
        name='{}\n'.format(model),
        text=['{}: {} '.format(error,error_value)
            for error_value in error_values]
        )

        return  plot_data 

def lookback(df_t,look_back,target_column): 
    data1, data2 = [], []

    for i in range(len(df_t)-look_back):
        a = df_t[i:(i+look_back)].to_numpy()
        data1.append(a)
        data2.append(df_t[target_column].loc[[i+look_back]].to_numpy())

    x_t, y_t = np.asarray(data1, dtype=np.float32), np.asarray(data2, dtype=np.float32)
    
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



def split_training_test_lstm(df, start_train,end_train,end_test):
#     # transform to datetime type
#     train_start =datetime.datetime.strptime(start_train, '%Y-%m-%d')
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






def lstm_predict(df_norm,folder_name, model_name,target_column, look_back, layers, epochs, start_train,end_train,end_test, verbose=0):
    

    # name to save the model
    today= datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    model_name='{}_lookback-{}_epochs-{}_{}'.format(model_name,look_back,epochs,today)
    
    # split trainig-test set
    df_train,df_test,diff_in_days= split_training_test_lstm(df, start_train,end_train,end_test)

    # save date index 

    
    # drop na and index
    df_train = df_train.dropna()
    df_train_index = df_train.index

    df_train = df_train.reset_index(drop=True)

    df_test = df_test.dropna()
    df_test_index = df_test.index    

    df_test = df_test.reset_index(drop=True)
    print(len(df_test_index))

     
    # TRAIN
    x_train, y_train = lookback(df_train,look_back,target_column)

    # TEST 
    x_test, y_test = lookback(df_test,look_back,target_column)

        
    # create model 
    model = create_LSTM_model(x_train,layers)

    # FIT
    history=model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=verbose)
    model.save('saved_model/{}/{}'.format(folder_name,model_name))

    # PREDICT
    train_predict = model.predict(x_train, batch_size=1)
    test_predict = model.predict(x_test, batch_size=1)

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



def plot_predictions(df, train_, test_,df_train_index,df_test_index, error,target_column):
    
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
        # extract dates 
        train_dates, test_dates=df_train_index,df_test_index
        
        
        train_predict= train_['predicted_values']
        y_train_ =train_['real_values']
        
        test_predict= test_['predicted_values']
        y_test_ =test_['real_values']
        
        
        data_plots=[]
        
        # error setting & computing
        if error == 'MEA':
            error_train=MEA(y_train_, train_predict)
            error_test=MEA(y_test_, test_predict)
        else:
            error_train= RMSE(y_train_, train_predict)
            error_test= RMSE(y_test_, test_predict)
            
        # train plotting
        # real values 
        plot_data=plot_Scatter(train_dates,y_train_, error, error_train,model='Real training')
        data_plots.append(plot_data)
        # predicted valus 
        plot_data=plot_Scatter(train_dates,train_predict,error, error_train,model='Predicted training')
        data_plots.append(plot_data)
        

        # plotting test 
        # real values 
        plot_data=plot_Scatter(test_dates,y_test_, error,error_train,model='Real test')
        data_plots.append(plot_data)
        #predicted values
        plot_data=plot_Scatter(test_dates,test_predict, error,error_train,model='Predicted test')
        data_plots.append(plot_data)




        # Create a layout with interactive elements and two yaxes
        layout = go.Layout(height=800, width=1000, 
                   font=dict(size=10),
                title='Predicted vs Real Values with {} '.format(error),
                   xaxis=dict(title='Date',
                                # Range selector with buttons
                                 rangeselector=dict(
                                     # Buttons for selecting time scale
                                     buttons=list([

                                         # Entire scale
                                         dict(step='all'),

                                         # 1 year
                                         dict(count=1,
                                              label='1 year',
                                              step='year',
                                              stepmode='backward'),
                                         # 6 month
                                         dict(count=6,
                                              label='6 months',
                                              step='month',
                                              stepmode='backward'),

                                         # 1 month
                                         dict(count=1,
                                              label='1 month',
                                              step='month',
                                              stepmode='backward'),
                                         # 1 week
                                         dict(count=7,
                                              label='1 week',
                                              step='day',
                                              stepmode='todate'),
                                         # 1 day
                                         dict(count=1,
                                              label='1 day',
                                              step='day',
                                              stepmode='todate'),

                                     ])
                                 ),
                                 # Sliding. for selecting time window
                                 rangeslider=dict(visible=True),
                                 # Type of xaxis
                                 type='date'),
                          yaxis=dict(title='{}'.format(target_column), color='black'),

                   )


        fig = go.Figure(data=data_plots, layout=layout)
        iplot(fig)
        
    



