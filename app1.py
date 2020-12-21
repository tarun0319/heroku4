#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from keras.models import load_model
import yfinance as yf
import datetime as dt
import streamlit as st
from PIL import Image
start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()



def main():
    
    image = Image.open('sm.jpeg')
    st.image(image,use_column_width=True)

    st.title("NSE Real-Time Stocks Analysis and Predictions")
    
    st.header("Select the stock and check its next day predicted value")
    
    st.subheader("This study is mainly confined to the s tock market behavior and is \
                intended to devise certain techniques for investors to make reasonable\
                returns from investments .")
    
    st.subheader("Though there were a number of studies , which\
                deal with analysis of stock price behaviours , the use of control chart\
                techniques and fai lure time analysis would be new to the investors. The\
                concept of stock price elast icity,\
                introduced in this study, will be a good\
                tool to measure the sensitivity of stock price movements.")
    
    st.subheader("In this study, \
                 Predictions for the close price is suggested for the National Stock Exchange index,\
                Nifty,\
                based on Long Short Term Based (LSTM)\
                method.") 
    
    st.subheader("We make predictions based on the last 30 days Closing price data\
                which we fetch from NSE India website in realtime.")
    
    st.markdown("Note: This is just a fun project, No one can predict the\
         stock market as of today because there are a\
        lot of factors which needs to be considered\
             before makaing any investments, especially in StockMarket.\
            So it is advisable now to indulge in any\
                 bad decisions based on the predictions shown here.")

    st.header("THANKS FOLKS!!")
    
    st.subheader("Happy Learning")

    st.subheader("Creator: MRINAL WALIA😈😈😈")

    st.header('Article Link: https://datascienceplus.com/real-time-national-stock-exchange-nse-of-india-close-price-stocks-predictions-in-python/')
    st.header('Source Code Link: https://github.com/abhiwalia15/AI-for-Finance-Stocks-real-time-analysis-')

    # Stock Section
    choose_stock = st.sidebar.selectbox("Choose the Stock!",
        ["NONE","Aditya Birla Fashion Retail Ltd.", "PowerMech Solns.", 'RepcoHomes', 'IndiaBulls HSG', 'INOX Leisure', 'SpiceJet', 'TataMotors'])

    if(choose_stock == "Aditya Birla Fashion Retail Ltd."):

        # get abfrl real time stock price
        ticker = 'WIPRO.NS'
        df1 = pdr.get_data_yahoo(ticker, start, end)
        df1['Date'] = df1.index

        st.header("Aditya Birla Fashion NSE Last 5 Days DataFrame:")

        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df1.tail())
        
        ## Predictions and adding it to Dashboard
        ## Predictions and adding it to Dashboard
        #Create a new dataframe
        new_df = df1.filter(['Close'])
        #Scale the all of the data to be values between 0 and 1 
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
        model = load_model("abfrl.model")
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        
        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df1[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High','Low']])



if __name__ == '__main__':

    main()
















        
