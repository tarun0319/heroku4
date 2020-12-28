import streamlit as st
PAGE_CONFIG = {"page_title":"Stock-Prediction-app","page_icon":"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxANEA8NDRAQDw0NEA0NDQ0PEBINDg4OFREWFhURFRUYHCggGRoxHRUTITEtJSk3Li4wFyI/ODM4NygtLysBCgoKDg0OGxAQGi0lICUtKy8vKy0rLS0yKy0vKy0tLS0uLy0tLS0tKy0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAOAA4AMBIgACEQEDEQH/xAAcAAADAAMBAQEAAAAAAAAAAAAAAQIDBgcEBQj/xABBEAABAwEGBAEJBQYFBQAAAAABAAIDEQQTITFRYQUGEkEiBxRScYGRobHBIzJCYnIkQ1OCkvAVMzRzokSDssPR/8QAGQEBAAMBAQAAAAAAAAAAAAAAAAECBAMF/8QAJBEAAgMAAgICAgMBAAAAAAAAAAECAxESMQQhImFBkTJRsUL/2gAMAwEAAhEDEQA/AO0MaWHqdlkvJxjiMNmidabRI2KGMUc95pj2aAMXHQDEo4pxSOzQyWi0uDIYWl73DPQNA7kkgAdyQvz5zZzJNxWYyyktiYT5vZ61ZC0/N57n6UC611ub+jnOzibhzL5WppaxcNjEMeXnEzQ+Z27Wfdb7a+xaJbePWy0Eme12iSubTM8M9jAQ0ewL59EUWyMIx6Rmcm+wzqTidTiUqBUBmlRWKioEUCdEUQehUCCMk6JkZICaBFAnRFEHoVAgAJ0TAQEkBFAqISogFQIoE6Iog9CAzRQKgM0qIBUCKBOiKIPQqBMYUIwOowKKJkZID32Lj1ss5BgtdojpkBM8s9rCS0+0Le+WvK1NF0xcSjE0eXnELRHM3dzPuu9lPauaURRVlCMu0WUmuj9PcH4jDaY22mzyNlheKNezXu0g4tOoOIXre0vxGWWi/OPKnMs3CphLES6J5HnFnrRkzB8njsfpUL9B8L4rFaYY7RZnB8MzQ9jjnoWkdiCCCOxBWOytwf0aYT5HLfLRx+9mj4dEfs4A2eemHVM4eBp9TTX+caLmi+nzHazaLZa5ia3lonI/QHkMH9IaPYvnUWyEeMUjLKWvSU06IorlQHdSrAzSogEkqoiiAlMjJOiZGSAmiSqiKICUwnRMBASUKiEqICaJ0ToiiAAM0lQGaVEBNE06IogEmeyKJkZICEUVURRATRdL8i/H7qaTh0p+znDp4K/hmaPG0etor/IdVzai+jy5ajZ7ZZJgaXdogJ/QXgPH9JcPaqTjyi0WjLHp4TjicziUqKqIorlSaIoqoiiAQGaVFYGaVFJBNEUVURRQSTRMjJOiZGSEEURRVRFEJJogBVRMBASQlRWQlRCCaIoqoiiEiAzSorAzSopIJoiiqiKKCSaJkZK2xuc0vDXFjSA54BLGk5AnIFIjJARRFFVEUQE0TGGIzGIToiiAqiKKqIopKk0RRVRFEAgM0qKwM0qICaIoqoiiAmiZGSdEyMkBFEUVURRATRACqiAEBJCKKiEUQE0SOC3jyf8AAuF8QddWqScWsVpZi9sUMrc6scB1E0zFQc+y6pw3lWwWWhgskLXDKRzL2Qfzvq74rjO5ReYdYVOS04Fw3g9ptf8AprPNMDk6ONzo/wCv7o962fh3kv4jNQyiGzNOd7J1vA2bGCPiF29C4vyJPo7Khfk5vw7yR2dtDarVLKRm2JrYGHbHqPuIWz8O5J4bZqGOyROcMQ+atodXUGQmnsWwoXJ2SfbOihFdI535Z5xHY7NZ20aJLR1dIwHRHG7Cnre1cgIyXRfLVaeq02SD+FBJKR/uyU/9S54RktlCyBktfzIoiiqiKLqcyaIoqoiiApCqiKKSCUKqJUQAO6SsDNKiEEoVURRCSUz2TomRkhBCFVEUQklAVUQAhBJQqIRRCSQSCCCQ5pDmuaS1zXA1BBGRXVORvKMH9Fk4m4NkwbFbDRrJD2bL2a782R7078tovbwbhE1vmZZrO3qkfiScGMYM3vPZor8u5C52QjJey8JuL9H6RQvn8A4WLDZobKJHyiFvTeSElzjnhXJvYDsAAvoLzmb0CEICA4R5TrVe8UtI7QiGAeyMOP8Ayc5auey9/HbTf2u1TVqJbRaHtP5TI7p+FF4iMl6cFkUjzpPZNkIVURRWIJQqovvcj8D/AMQtsULhWGM39o0umEeH2ktb6idFEmktZKWvD4dEqK6IorFNJQqoiiDRDukrAzSog0iiaqiKINJTIyRRURkg0hCqiKINJQFVEAINJKFRCzWGxyWiRkEDDJLKeljBmT9B3J7AKGB8M4dLa5WWezs65ZDRoyAHdzj2aO5XduUeWYuFw3bPHM+jrRORQyOHYaNGNB9SSsXJnKsfC4qYPtMoF/Nr+RmjB8cztsSw3W8vS6N1VXH2+wWl+ULnIcPZ5tZiDbpW54EWZh/eEeloPacMD7eeOa2cMi6WUfbJQbiI4hgyvX/lHYdz7SOHzzPle+WVzpJZHF8kjjVznHMlTTTy9vordbx9Ls6vyL5QG2noslvLWWrBsU+DY7QewPZr/ge1MluvF7V5vZ7ROf3MM0v9LCfovzeWg4HELaoed5zYLRw60VmvIxFBOTWRreodTJCfvDp6qHPWva9nj+9iUhf6yRqTW0AGgAVnsnRMjJazNpCFvHKvIn+IWCa0l3RO95FjJP2Zayod1DQuqK9uiuoOm2mzPhe+KVjo5Y3FkkbsHNcO395qkZptpfgs4tJN/kwrqnKUbeC8Im4lKPt7U1skbTmQfDZ2eolxednbLROU+CniFrhs1Ddk3k50gbQv9+DfW4LavKrxtr547AwfZ2QB8gFOm+c3wtpsw/8AM6Lnb8mofs6V/FOf6NAoiiuiKLuZtIoiiuiKINEBmposoGazWGxSWmRkEDDJLIeljB33Og7k9kYTPJRFF2vgnIdlgsrrPaWNnlnAv5si1wyERzaBrme+g51zdyhNwx3VjLZHGkc4GLScmSAfddvkfgOULoyeHadMorWazRURgE6KiMl2OOmKiKK6J0UDTHRMDJXRXDE57msY0ue5zWta0Vc5xNAAO5QaTBZ3yvbFE1z5JHdDGNFXOcewXa+R+UmcMj65KPtkrReyDERtzumbanuRsAMXIvKDeHtv5wHW2QeI5iBh/dtOup7+rPbVhuu5el0bqKePyl2C+FzdzLFwuG8dR88lW2eCtDI7U6NGFT9SF6eYuORcOgdaJjX8MUQPjlkpgxv1PYLhnGuJy26Z9ptBq9+AaPuxsGUbdAK/M5lVpq5vX0Wuu4LF2ey18Lt9shl4vMx0rJHnrk/F0j8TWdohlhlTQVXwqLq3InPLJGx2G29MUjQ2KCYAMikAFGscBgx2Q0OxwMc6eT0P6rVw5obJi6WyDwtfq6L0XbZHtTvpVvGXGSz+jO6uUeUHv9nLKIosjmFpLXAhzSWuaQQ5rgaEEHIpUXcz6RReixWN9okigiFZJntiZ3ALjSp2GZ2CxUXQPJNwe8mktzx4bO0xRf7zx4j7G4f9xVnLjFstXHnJI2vmLjUXAbLZIo47wB0cDIq9LjCxv2j665e1wXg5o4DBx2zMt9gc02gN+zd92+aM4ZNHA1pXI4ZFah5Q+Mi1W6Ro8Udl/Z2ZEdQNZD/VUfyBeTlTml/DJepoL7PIRfwCni7dbdHj45HsRmjVJRUl2aZXRcnF9G28k2McH4fauJ2phbM8OpG8dDwxji1kdDkXPP8A46LnFothle+WTxSSudJI6gxe41J95W8eUjmmK1iz2ezPEkBa20yObkXmoYwjsQOokHuRotGvW+j8AutUW9k+2crZJZFdIx0RRVRFF3Muk0RRVRZrHZJJ5GQwsMksh6WMbmT9B3r2og0LDY5LRI2GFhklkIaxjcyfoO5PZdo5O5Vj4ZHU0fapAL6bsBnds0b88z2AXJ3KsfDI6uo+1yAX0wyaM7tmjfn7gNjWC67l6XR6VFHD5S7BRPC2Rro5Gtex4LXscA5rmnMEHMK1Mbw4BzSHNcA5rmkOa4HIgjMLOaTk/OnIbrL1WmxB0lmxdJFi6SAaju5nxHeoxGkkYBfpBaBznyGJeq02BobLi6SzCjWSHMuj7Ndtkdjnsp8j8S/Zhv8AG/6h+jltEUWVzC0lrgQ5pLXNcC1zSMwQcil0rYYdIawkgAEkkAAAkknAADuV1zkLk4WJotVpaDbHjwswIszT2H5yMz2yHcnDyDyd5sG221t/aSKwxOH+nB/ER6fy9dVvSw33b8YnoePRnykC8XF+JxWKF9otDumNg9bnuOTGju4rNbrXHZ43zzODIowXPeew+p7Ad1xnmjmV/EpetwLYIyRZ4SR4R6btXn4ZbnlVU5v6O19yrX2eTmDj8nEJjPMKAVbDFm2GP0RvlU9/cB828b6PyWS8Ho/JF4PR+S9FRSWI8tzbetmPraQQW4exbtydz6bP02a3FzrPg2O0HxPhHYP7uZ8RuMtOEgx8PyU3g9H5Ks4KSxkwtcHqZ1vmzlCDijBaIHMjtJaHMnbjFO2mAfTMUpRwxG4wXIuI8PlssjoLQwxyszae47OByI3C2PlTnCThxEbg6WyE1dDXxR1zdHXL1ZHbNdH4hw+x8cszXgh7CCYbRHhLC7uMct2lZ1KVLyXRrcY3rY+mcMDScACScABiSdAu0MA4FwnGl7FHU9+u1yHLcdTqepq1jlzkeeDiLBaWh1ns9bQyZo+zmc0/ZjZ3UQ4g+j3FCcvlW4uDJDYm4iMecS/rcCGD2N6j/OFNjVklFddla06oSnLvo0C9GZFScSTiSdSlet9H5Kr1vo/JF630Vpwy6TeN9H4BF630fkqvW+ii9b6KYNMaE1ls1nfM9kUTS+SQhrGNxc52n95KxQVlsz5nsiiaXyyENYxuJcf79y7LybyqzhsfU6j7XIKSyjEMH8Nn5dT39wC5M5UZw1nW+j7ZIKSyDEMbnds21PenqA2Veffdy9Lo9Px/H4fKXf8AgIQtN595t8zHmlmP7XI3xvH/AE8Z7/rPbTPSvGMXJ4jROahHkz5vlC5uoX8PsrscW2uVpy1hadfS0y1prnK3N0vDiGUMtkJq6Coqyuboycjtkds18ESjT61TEo0XoRpio8cPJl5EnPkmd44VxKG2RNns7xJG7uMC13drhmDsV7Fwrg3G5rDJfWc0JoJIzjHK3Rw+uYXW+WeZYOJMrH4JmistncfGzcek3f30WO2lw9/g30eTGz0+zw838nx8QBli6YrYBhJTwS0ybJT55jcYL5XI/JRgd53bmC+Y43EFQ4RkH/NcRgTppnnSm+oVVbJR46dHRBz557BRNK2NrnvcGsYC573GjWtAqSSqJpUk0AxJOAAXKudObfPX+b2cnzNjhVww84eD979AOWuelFdbm8QuujVHWePm/mo8Qk6GAtskR+yYcDI7+K4fIdvWVr14NPkrMg0RejRelGCisR407HN62ReD0fki8GnyV3g0SvRorYV0QkGOCm8GnyWQSChwU3o0TBpF4NF9LgPMM3D5LyDFriL2Fx+zlG+h0I+WC8F6NEr0aKrimsZaM2nqZ2rg3MtmtkD7Sx/S2FpfaI30EkAAJPUNMDQ5Gi45xTiRtU0lpePFM9zyDSrR+FvsAA9iwNtBbXpq3qaWO6TTqYc2mmYwCTpRhguddKg2zvb5DsSTIvRp8kr0afJVejRF6NF1OOk3o0+SL0afJO9GiV8NEJ0VngfK9sUTS+SQhrGNFXOcewXYuS+U2cOZeSUfbJG0kkGLY2/w2banv6qBLkrlJnDmXstH2yRtHvGLYmn92z6nv6ltCw33cviuj0fH8fj8pd/4CEL4/M/H4+GwmV/ikdVsENaOlf8ARozJ7eshZ0m3iNUpKK1nk5z5nbw6LpZR1rlBuYziGDIyu2HbU+2nIJLSXuc95L3vJc97jVznHMkrNbOIvtMkk8565ZDVzu2zQOwGQCwCUaL0aqlBfZ4997sl9CEo0VXo0RejRMSDRdjPoCUaLLBa3ROZLEXRysNWPaaOaVjvBoqMgwwTCOWfk6fyjzqy2dNntNIrVk05Rz/p0dt7tBt64AXg9l993OVrNmNkc6tfD5xU311TFhPc75096x2eL7+J6FPnJLJn1ee+bRMXWKyurC0ltolaf80942n0ddfVnpjZBomHgYUTEg0WqutQWIw23OyXJkmQaIvBoqMg0ReDRXw56TeDRK8Giu8GiV4NEwnkISDHBTeDRZBIMcFN4NFA0i8GiV6NFd4NErwaITpF6NE3SjDBMyDRDpRhgmE6Y70aJXo0V3o0UmUaKC2ivRokZRonejRF6NELafoZCFhttrjgjfNM4MijaXPecgPqfmvIPdPPxnisVhhfaJzRjMA0fekecmNHcn+8AuL8a41JbpnWibM4MjBq2KPsxv1Pcr08z8yP4jNeOBbBHUWeEn7rfSd+Y99MvX8kSjRehRTwWvs8nyfI5vF0NsoxwTvRohsgxwTEmy0GTQEo0TvBoi82VCTZThVsV4NFRkGGCBINFRkywQq2TebJ3g0TvBonebJhGivBomJNk7zZMSbJhGkmTZK82VmTZK82U4NJvNkXg0VXmyLzZRg0kSDHBTebLJeZ4KbzZCdIMg0SvBorMg0UmTZCyZN4NEOkGGCZk2Q6TLBQTpjvRolejRVeDRIybIWTIvRokZRoqMo0SMo0UFtP0G94aC5xDWtBc5xNA0DEknsFyDnPms8QkuoqixxO8Ayvnj964aaD25nD38/c3ecOdYrK79nYaTyA/wCe8fgH5Afedhjpol2WXx6c+Ujb5Xkb8IjEuyYk2SEuyoSbLWYGymyZ4IEmybZM8ExJspKaAk2VCTZAk2TEmyFdASbKjJlggSbKjJlgpKtk3myfXsn17KuvZCuk9eyYk2T69kw/ZThGkmTZF5srL9kuvZMGkXmyOvZX17JdeyYTpIfngpvNlk688FPXsowlMgybKTJsshfspMmyYW0gybJOkywVmTZJ0mWCgnTGZNlJk2VmTZSZNkLpkmXZSZdlRk2SvNlBZM9fGori02iEil3NKB+nqJafcQvIJdlvXlP4RdyMtsbaslAimp+GQDwuPrGH8u60USbKlcuUUzrdFwm4jEmyoSbJCTZUJNl0ODZTZM8ExJshsmeCYk2+KFGxiTZUJNkCTZUH7fFSVbAP2VdeWCQfsr68sFOFGxCTZMP2TD9kw/ZCuivNkw/ZPr2+KYfspI0kv2R17Ki/ZHXshGkdeyLzZX17Jde3xUE6R154KbzZZOvPBSX7ISmQZNlJfsshfspL9lGF0zGZNkOkywVF+yTpMsELJmMybKTJsrMmykybfFQXTIMmy9fBYr+02eICvXNED+nqBcfcCvKZNlvHkx4TeSPtsg8EQMUNfxSEeJw9Qw/m2XOyXGLZ3pjzmoo3y22Bk0b4pwHRSNLXAZ7EaGuK49zFwWXh8vQ8F0TibmbJr26HR2oXaGOLzQ5LBxGyRysMMrGvjcKlrscdQexXn03Ot/R63keOrVq7OFCTZUJNlvXG/J/IysljeHszupD0vGwdkfbRaraeFWqGoks8rad+glvvAovRhZGXTPHsqnD+SPG2TPBMSbKmh2PgPuP/AMTAd6J9xXTDO5L+xB+yoSbKh1eifcVQr6J+KnCjkSH7K+vLBAr6J+Kupw8JU4UbID9lQfsqBOhTqdEwq2T17Jh+yqp0TBOinCukl+yXXsshJ0SqdEwaR17JdeyydR0SqdFGE6Y+vPBSX7fFZanHBSSdCmEpmIv2UmTZZST6JSNfRPxTC6ZhMmyTpMsPisp6vRPuKR6sPCfcVGFlIwGTb4qTJsvo2bhVpmNI7PK6vfpIb7yKLaeC8gyPo+2PDGZ3UZ6nnYuyHsquU7Ix7ZpqqnZ/FGscvcGlt8gYwFsTSL2bNrBoNXaBdesVgZDGyKABsUbQ1oOe5Op7o4dZI4mCGJjWRtFQ1uGOpPcrO9xZgMs9V511zsf0ez43jKpe+z//2Q==","layout":"centered"}
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
from keras.models import model_from_yaml
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
   
    
    
