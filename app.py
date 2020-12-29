import streamlit as st
PAGE_CONFIG = {"page_title":"Stock-Prediction-app","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import PIL





import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import streamlit as st
import pandas_datareader as pdr

from PIL import Image
import datetime as dt
start = dt.datetime(2018, 1, 1)
end = dt.datetime.now()
from keras.models import load_model

#Scale the all of the data to be values between 0 and 1 
st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
  st.title("NSE Real-Time Stocks Analysis and Predictions")
  image = Image.open('Stock_Data/imgq/stk.jpg')
  st.image(image,use_column_width=True)
  st.header("Select the stock and check its next day predicted value")
  # Stock Section
  choose_stock = st.sidebar.selectbox("Choose the Stock!",
  [ "None", 'ADANIPORTS.NS','ASIANPAINT.NS','AXISBANK.NS','BAJAJ-AUTO.NS','BAJFINANCE.NS','BAJAJFINSV.NS','BHARTIARTL.NS','BPCL.NS','BRITANNIA.NS','CIPLA.NS',
'COALINDIA.NS','DIVISLAB.NS','DRREDDY.NS','EICHERMOT.NS','GAIL.NS','GRASIM.NS','HCLTECH.NS','HDFC.NS','HDFCBANK.NS','HEROMOTOCO.NS','HINDALCO.NS',
'HINDUNILVR.NS','ICICIBANK.NS','INDUSINDBK.NS','INFY.NS','IOC.NS','ITC.NS','JSWSTEEL.NS','KOTAKBANK.NS','LT.NS','M&M.NS','MARUTI.NS','NESTLEIND.NS',
'NTPC.NS','ONGC.NS','POWERGRID.NS','RELIANCE.NS','SBIN.NS','SHREECEM.NS','SUNPHARMA.NS','TATAMOTORS.NS','TATASTEEL.NS','TCS.NS','TECHM.NS','TITAN.NS',
'ULTRACEMCO.NS','UPL.NS','WIPRO.NS'])
  
  
  if (choose_stock == "ADANIPORTS.NS"):
    # get abfrl real time stock price
    ticker = 'ADANIPORTS.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/ADANIPORTS.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/ADANIPORTS.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
######################################################################2
  elif (choose_stock == "ASIANPAINT.NS"):
    # get abfrl real time stock price
    ticker = 'ASIANPAINT.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/ASIANPAINT.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/ASIANPAINT.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
################################################################################################################3
  elif (choose_stock == "AXISBANK.NS"):
    # get abfrl real time stock price
    ticker = 'AXISBANK.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/AXISBANK.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/AXISBANK.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
#######################################################################################4
  elif (choose_stock == "BAJAJ-AUTO.NS"):
    # get abfrl real time stock price
    ticker = 'BAJAJ-AUTO.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/BAJAJ-AUTO.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/BAJAJ-AUTO.NSmodel.h5")
    a = []
    b= []
  
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
########################################################################################5
  elif (choose_stock == "BAJAJFINSV.NS"):
    # get abfrl real time stock price
    ticker = 'BAJAJFINSV.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/BAJAJFINSV.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/BAJAJFINSV.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
################################################################################6
  elif (choose_stock == "BAJFINANCE.NS"):
    # get abfrl real time stock price
    ticker = 'BAJFINANCE.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/BAJFINANCE.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/BAJFINANCE.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
##########################################################################################################7
  elif (choose_stock == "BHARTIARTL.NS"):
    # get abfrl real time stock price
    ticker = 'BHARTIARTL.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/BHARTIARTL.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/BHARTIARTL.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
#######################################################################################################8
  elif (choose_stock == "BPCL.NS"):
    # get abfrl real time stock price
    ticker = 'BPCL.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/BPCL.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/BPCL.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
##############################################################################################9
  elif (choose_stock == "BRITANNIA.NS"):
    # get abfrl real time stock price
    ticker = 'BRITANNIA.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/BRITANNIA.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/BRITANNIA.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
########################################################################################10

  elif (choose_stock == "CIPLA.NS"):
    # get abfrl real time stock price
    ticker = 'CIPLA.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/CIPLA.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/CIPLA.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
#######################11
  elif (choose_stock == "COALINDIA.NS"):
    # get abfrl real time stock price
    ticker = 'COALINDIA.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/COALINDIA.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/COALINDIA.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
########################12
  elif (choose_stock == "DIVISLAB.NS"):
    # get abfrl real time stock price
    ticker = 'DIVISLAB.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/DIVISLAB.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/DIVISLAB.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
###################13
  elif (choose_stock == "DRREDDY.NS"):
    # get abfrl real time stock price
    ticker = 'DRREDDY.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/DRREDDY.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/DRREDDY.NSmodel.h5")
    a = []
    b= []
  
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
############14
  elif (choose_stock == "EICHERMOT.NS"):
    # get abfrl real time stock price
    ticker = 'EICHERMOT.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/EICHERMOT.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/EICHERMOT.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
#####15
  
  elif (choose_stock == "GAIL.NS"):
    # get abfrl real time stock price
    ticker = 'GAIL.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/GAIL.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/GAIL.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
############16
  elif (choose_stock == "GRASIM.NS"):
    # get abfrl real time stock price
    ticker = 'GRASIM.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/GRASIM.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/GRASIM.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
###################17
  elif (choose_stock == "HCLTECH.NS"):
    # get abfrl real time stock price
    ticker = 'HCLTECH.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/HCLTECH.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/HCLTECH.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
##################18
  elif (choose_stock == "HDFC.NS"):
    # get abfrl real time stock price
    ticker = 'HDFC.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/HDFC.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/HDFC.NSmodel.h5")
    a = []
    b= []
 
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
################19
  elif (choose_stock == "HDFCBANK.NS"):
    # get abfrl real time stock price
    ticker = 'HDFCBANK.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/HDFCBANK.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/HDFCBANK.NSmodel.h5")
    a = []
    b= []
  
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
########20
  elif (choose_stock == "HDFCLIFE.NS"):
    # get abfrl real time stock price
    ticker = 'HDFCLIFE.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/HDFCLIFE.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/HDFCLIFE.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
##################21
  elif (choose_stock == "HEROMOTOCO.NS"):
    # get abfrl real time stock price
    ticker = 'HEROMOTOCO.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/HEROMOTOCO.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/HEROMOTOCO.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
#######22
  elif (choose_stock == "HINDALCO.NS"):
    # get abfrl real time stock price
    ticker = 'HINDALCO.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/HINDALCO.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/HINDALCO.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
##########23
  elif (choose_stock == "HINDUNILVR.NS"):
    # get abfrl real time stock price
    ticker = 'HINDUNILVR.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/HINDUNILVR.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/HINDUNILVR.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
######24
  elif (choose_stock == "ICICIBANK.NS"):
    # get abfrl real time stock price
    ticker = 'ICICIBANK.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/ICICIBANK.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/ICICIBANK.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
##########25
  elif (choose_stock == "INDUSINDBK.NS"):
    # get abfrl real time stock price
    ticker = 'INDUSINDBK.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/INDUSINDBK.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/INDUSINDBK.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
###########26
  elif (choose_stock == "INFY.NS"):
    # get abfrl real time stock price
    ticker = 'INFY.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/INFY.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/INFY.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
###27
  elif (choose_stock == "IOC.NS"):
    # get abfrl real time stock price
    ticker = 'IOC.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/IOC.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/IOC.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
#####28
  elif (choose_stock == "ITC.NS"):
    # get abfrl real time stock price
    ticker = 'ITC.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/ITC.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/ITC.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####29
  elif (choose_stock == "JSWSTEEL.NS"):
    # get abfrl real time stock price
    ticker = 'JSWSTEEL.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/JSWSTEEL.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/JSWSTEEL.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
##30
  elif (choose_stock == "KOTAKBANK.NS"):
    # get abfrl real time stock price
    ticker = 'KOTAKBANK.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/KOTAKBANK.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/KOTAKBANK.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####31
  elif (choose_stock == "LT.NS"):
    # get abfrl real time stock price
    ticker = 'LT.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/LT.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/LT.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####32
  elif (choose_stock == "M&M.NS"):
    # get abfrl real time stock price
    ticker = 'M&M.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/M&M.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/M&M.NSmodel.h5")
    a = []
    b= []
  
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
###33
  elif (choose_stock == "MARUTI.NS"):
    # get abfrl real time stock price
    ticker = 'MARUTI.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/MARUTI.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/MARUTI.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####34
  elif (choose_stock == "NESTLEIND.NS"):
    # get abfrl real time stock price
    ticker = 'NESTLEIND.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/NESTLEIND.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/NESTLEIND.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####35
  elif (choose_stock == "NTPC.NS"):
    # get abfrl real time stock price
    ticker = 'NTPC.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/NTPC.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/NTPC.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####36
  elif (choose_stock == "ONGC.NS"):
    # get abfrl real time stock price
    ticker = 'ONGC.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/ONGC.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/ONGC.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####37
  elif (choose_stock == "POWERGRID.NS"):
    # get abfrl real time stock price
    ticker = 'POWERGRID.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/POWERGRID.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/POWERGRID.NSmodel.h5")
    a = []
    b= []
  
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####38
  elif (choose_stock == "RELIANCE.NS"):
    # get abfrl real time stock price
    ticker = 'RELIANCE.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/RELIANCE.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/RELIANCE.NSmodel.h5")
    a = []
    b= []
    
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####39
  elif (choose_stock == "SBIN.NS"):
    # get abfrl real time stock price
    ticker = 'SBIN.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/SBIN.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/SBIN.NSmodel.h5")
    a = []
    b= []
  
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
#####40
  elif (choose_stock == "SBILIFE.NS"):
    # get abfrl real time stock price
    ticker = 'SBILIFE.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/SBILIFE.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/SBILIFE.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####41
  elif (choose_stock == "SHREECEM.NS"):
    # get abfrl real time stock price
    ticker = 'SHREECEM.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header( "Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/SHREECEM.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/SHREECEM.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####42
  elif (choose_stock == "SUNPHARMA.NS"):
    # get abfrl real time stock price
    ticker = 'SUNPHARMA.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/SUNPHARMA.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/SUNPHARMA.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
###43
  elif (choose_stock == "TATAMOTORS.NS"):
    # get abfrl real time stock price
    ticker = 'WIPRO.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/TATAMOTORS.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/TATAMOTORS.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####44
  elif (choose_stock == "TATASTEEL.NS"):
    # get abfrl real time stock price
    ticker = 'TATASTEEL.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/TATASTEEL.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/TATASTEEL.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()

####45
  elif (choose_stock == "TCS.NS"):
    # get abfrl real time stock price
    ticker = 'TCS.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/TCS.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/TCS.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####46
  elif (choose_stock == "TECHM.NS"):
    # get abfrl real time stock price
    ticker = 'TECHM.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/TECHM.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/TECHM.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####47
  elif (choose_stock == "TITAN.NS"):
    # get abfrl real time stock price
    ticker = 'TITAN.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/TITAN.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/TITAN.NSmodel.h5")
    a = []
    b= []
   
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()

###48
  elif (choose_stock == "ULTRACEMCO.NS"):
    # get abfrl real time stock price
    ticker = 'ULTRACEMCO.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header("Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/ULTRACEMCO.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/ULTRACEMCO.NSmodel.h5")
    a = []
    b= []
  
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####49
  elif (choose_stock == "UPL.NS"):
    # get abfrl real time stock price
    ticker = 'UPL.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/UPL.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/UPL.NSmodel.h5")
    a = []
    b= []
  
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()
####50
  elif (choose_stock == "WIPRO.NS"):
    # get abfrl real time stock price
    ticker = 'WIPRO.NS'
    df1 = pdr.get_data_yahoo(ticker, start, end)
    df1['Date'] = df1.index

    st.header(" Last 10 Days DataFrame:")

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
      st.subheader("Showing raw data---->>>")	
      st.dataframe(df1.tail(10))

    st.write('----')
    ## Predictions and adding it to Dashboard
    ## Predictions and adding it to Dashboard
    #Create a new dataframe
   
    new_df = df1.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    scaled_data = scaler.fit_transform(new_df)
    #Get teh last 30 day closing price 
    last_30_days = new_df[-30:].values
    #Scale the data to be values between 0 and 1
    last_30_days_scaled = scaler.transform(last_30_days)
    #Create an empty list
    X_test = []
    #Append teh past 1 days
    X_test.append(last_30_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #load the model
    model = load_model("Stock_Data/WIPRO.NSmodel.h5")
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
        
    # next day
    NextDay_Date = dt.date.today() + dt.timedelta(days=1)

    st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    st.write(pred_price)

    st.write('----')

    ##visualizations
    st.subheader("Close Price VS Date Interactive chart for analysis:")
    st.line_chart(df1['Close'])

    st.write('----')

    st.subheader("Line chart of Open and Close for analysis:")
    st.area_chart(df1[['Open','Close']])
    st.write('----')
    # prediction for next 30 dayes
   

    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    x_input=new_df[-100:].values
    x_input = scaler.fit_transform(x_input)

    x_input= x_input.reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    model = load_model("Stock_Data/WIPRO.NSmodel.h5")
    a = []
    b= []
 
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
      if (len(temp_input)>100):
    #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
    #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        
        a.append(i)
        b.append(yhat)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
    #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
      
      else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
    
    st.subheader("predictione for next 30dayes:")
    asd = scaler.inverse_transform(lst_output)
    st.write(asd)
    st.write('----')
    st.subheader("Graph for next 30dayes:")
  
    plt.plot(asd)
    st.pyplot()
    st.write('----')
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.figure(figsize=(16,8))
    plt.title('Curve Of Predictione')
    plt.xlabel('Dayes', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(day_new,new_df[-100:])
    plt.plot(day_pred,asd)
    plt.show()
    st.pyplot()

 

   
    


    
   
        

         
       
        


          


        




        
if __name__ == '__main__':

    main()
   
    
    
