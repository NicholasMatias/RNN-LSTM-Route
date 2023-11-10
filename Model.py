import pandas as pd 
import matplotlib.pyplot as plt
import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

US_Stocks = pd.read_csv("US-India-Technology Sector\\^NDXT-SEPT.16.2007-NOV.05.2023.csv")
India_Stocks = pd.read_csv("US-India-Technology Sector\\Nifty IT-SEPT.16.2007-NOV.05.2023.csv")

#print("US Dataframe:\n",US_Stocks)
#print("India Dataframe:\n",India_Stocks)

US_Stocks = US_Stocks[['Date','Close']]
US_Stocks['Close']=US_Stocks['Close']/100.000
India_Stocks = India_Stocks[['Date','Close']]
India_Stocks['Close']= India_Stocks['Close']/83.321047

#print("US Updated Dataframe:\n",US_Stocks)
#print("India Updated Dataframe:\n",India_Stocks)
def check_Date(date):
    dateFormat ='%Y-%m-%d'
    if(len(date)!=10):
        raise ValueError("Invalid date entered, should be YYYY-MM-DD")
    try: 
        dateObject = datetime.datetime.strptime(date,dateFormat)
    except ValueError:
        print("Incorrect data format, should be: YYYY-MM-DD")
    if date not in US_Stocks['Date'] or date not in India_Stocks['Date']:
        raise ValueError('Date not in dataframe, should be between 2007-09-17 and 2023-11-03\nThere are a few exceptions as the Indian stocks dataset is missing a few dates. ')


def str_to_datetime(string):
    split = string.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year= year, month=month, day=day)

US_Stocks['Date'] = US_Stocks['Date'].apply(str_to_datetime)
India_Stocks['Date'] = India_Stocks['Date'].apply(str_to_datetime)

#print(US_Stocks['Date'])
#print(India_Stocks['Date'])

US_Stocks.index = US_Stocks.pop('Date')
India_Stocks.index = India_Stocks.pop('Date')

#print(US_Stocks)
#print(India_Stocks)

userStartDate='2021-03-25'
userEndDate='2023-11-03'
epochsCount=100
'''userStartDate= input('Enter the start date: ')
check_Date(userStartDate)
userEndDate= input('Enter the end date: ')
check_Date(userEndDate)'''


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date  = str_to_datetime(last_date_str)

    target_date = first_date
    
    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        
        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return 

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
        
        if last_time:
            break
        
        target_date = next_date

        if target_date == last_date:
            last_time = True
        
    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
    
    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n-i}'] = X[:, i]
    
    ret_df['Target'] = Y

    return ret_df



# Start day second time around: '2021-03-25'
US_windowed_df = df_to_windowed_df(US_Stocks, 
                                userStartDate, 
                                userEndDate, 
                                n=3)
India_windowed_df = df_to_windowed_df(India_Stocks, 
                                userStartDate, 
                                userEndDate, 
                                n=3)
#print(US_windowed_df)
#print(India_windowed_df)

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


US_dates, US_X, US_Y = windowed_df_to_date_X_y(US_windowed_df)
India_dates, India_X, India_Y = windowed_df_to_date_X_y(India_windowed_df)
#print(f'Dates shape: {US_dates.shape}   US X shape: {US_X.shape}   US Y shape: {US_Y.shape}')
#print(f'Dates shape: {India_dates.shape}   India X shape: {India_X.shape}   India Y shape: {India_Y.shape}')



q_80 = int(len(US_dates)* .8)
q_90 = int(len(US_dates)* .9)

US_dates_train, US_X_train, US_Y_train = US_dates[:q_80], US_X[:q_80], US_Y[:q_80]
US_dates_val, US_X_val, US_Y_val = US_dates[q_80:q_90], US_X[q_80:q_90], US_Y[q_80:q_90]
US_dates_test, US_X_test, US_Y_test = US_dates[q_90:], US_X[q_90:], US_Y[q_90:]


plt.plot(US_dates_train, US_Y_train)
plt.plot(US_dates_val, US_Y_val)
plt.plot(US_dates_test, US_Y_test)

plt.title('US Stock Data')

plt.legend(['Train','Validation','Test'])

plt.show()

India_dates_train, India_X_train, India_Y_train = India_dates[:q_80], India_X[:q_80], India_Y[:q_80]
India_dates_val, India_X_val, India_Y_val = India_dates[q_80:q_90], India_X[q_80:q_90], India_Y[q_80:q_90]
India_dates_test, India_X_test, India_Y_test = India_dates[q_90:], India_X[q_90:], India_Y[q_90:]

plt.plot(India_dates_train, India_Y_train)
plt.plot(India_dates_val, India_Y_val)
plt.plot(India_dates_test, India_Y_test)

plt.title('India Stock Data')

plt.legend(['Train','Validation','Test'])


plt.show()



US_model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

US_model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

US_model.fit(US_X_train, US_Y_train, validation_data=(US_X_val, US_Y_val), epochs=epochsCount)




India_model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

India_model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

India_model.fit(India_X_train, India_Y_train, validation_data=(India_X_val, India_Y_val), epochs=epochsCount)



India_train_predictions = India_model.predict(India_X_train).flatten()


plt.plot(India_dates_train, India_train_predictions)
plt.plot(India_dates_train, India_Y_train)
plt.legend(['India Training Predictions', ' India Training Observations'])
plt.show()





US_train_predictions = (US_model.predict(US_X_train).flatten())*100

plt.plot(US_dates_train, US_train_predictions)
plt.plot(US_dates_train, US_Y_train*100)
plt.legend(['US Training Predictions', ' US Training Observations'])
plt.show()



India_val_predictions = India_model.predict(India_X_val).flatten()


plt.plot(India_dates_val, India_val_predictions)
plt.plot(India_dates_val, India_Y_val)
plt.legend(['India Value Predictions', ' India Value Observations'])
plt.show()


US_val_predictions = US_model.predict(US_X_val).flatten()*100


plt.plot(US_dates_val, US_val_predictions)
plt.plot(US_dates_val, US_Y_val*100)
plt.legend(['US Value Predictions', ' US Value Observations'])
plt.show()


India_test_predictions = India_model.predict(India_X_test).flatten()


plt.plot(India_dates_test, India_test_predictions)
plt.plot(India_dates_test, India_Y_test)
plt.legend(['India Test Predictions', ' India Test Observations'])
plt.show()



US_test_predictions = US_model.predict(US_X_test).flatten()*100


plt.plot(US_dates_test, US_test_predictions)
plt.plot(US_dates_test, US_Y_test*100)
plt.legend(['US Test Predictions', ' US Test Observations'])
plt.show()