# inmporting libraries
import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_datareader.data as web
import CAPM_functions

st.set_page_config(page_title='CAPM',page_icon="chart_with_upward_trend",layout='wide')
st.title("Capital Asset Pricing Model")

#getting input from user

col1,col2=st.columns([1,1]) # [1,1] represent equal sizes for both the columns
with col1:
    stocks_list=st.multiselect("Choose 4 stocks",('TSLA','AAPL','NFLX','MSFT','MGM','AMZN','NVDA','GOOGL'),['TSLA','AAPL','NFLX','NVDA'])
with col2:
    year= st.number_input("Number of years",1,10)# 1 is the default value and 10 is the maximum acceptable value

# downloading data from SP500

try:
    end= datetime.date.today()
    start=datetime.date(datetime.date.today().year-year,datetime.date.today().month,datetime.date.today().day)
    SP500= web.DataReader(['SP500'],'fred',start,end) # fred -> federal reserve economic data
    #print(SP500.head())

    stock_df=pd.DataFrame()

    for stock in stocks_list:
        data_=yf.download(stock,start=start,end=end )# f string evaluates the value inside the string literal and put that value in the string 
        # print(data_.head()) # data contains high,low,open,close,volumn,adjacent close 
        stock_df[f'{stock}']=data_['Close']

    #print(stock_df.head())

    stock_df.reset_index(inplace=True)
    SP500.reset_index(inplace=True)

    #print(stock_df.dtypes)
    #print(SP500.dtypes)

    SP500.columns=['Date','sp500'] # changing DATE to Date to join SP500 with stock_df

    stock_df=pd.merge(stock_df,SP500,on= 'Date',how='inner')
    #print(stock_df)

    #col1,col2= st.columns([1,1])
    with col1:
        st.markdown("Dataframe head")
        st.dataframe(stock_df.head(),use_container_width=True)
    with col2:
        st.markdown("Dataframe tail")
        st.dataframe(stock_df.tail(),use_container_width=True)

    col1,col2=st.columns([1,1])
    with col1:
        st.markdown("Price of all the stocks")
        st.plotly_chart(CAPM_functions.interactive_plot(stock_df))
    with col2:
        st.markdown("Normalized price of all the stocks")
        st.plotly_chart(CAPM_functions.interactive_plot(CAPM_functions.normalize(stock_df)))
    #print(stock_df)
    stocks_daily_return=(CAPM_functions.daily_return(stock_df))

    beta={}
    alpha={}

    for i in stocks_daily_return.columns:
        if i!='Date' and i!='sp500':
            b,a=CAPM_functions.calculate_beta(stocks_daily_return,i)

            beta[i]=b
            alpha[i]=a
    print(beta,alpha)

    beta_df=pd.DataFrame(columns=['Stock','Beta Value'])
    beta_df['Stock']=beta.keys()
    beta_df['Beta Value']=beta.values()

    with col1:
         st.markdown('Calculated Beta Values')
         st.dataframe(beta_df,use_container_width=True)

    rf=2 # risk free return
    rm= stocks_daily_return['sp500'].mean()*252 # average yearly return of sp500

    return_df=pd.DataFrame(columns=['Stock','Return Value'])
    for i in range(0,len(beta_df)):
        return_df.loc[i, 'Stock'] = beta_df['Stock'][i]
        return_df.loc[i, 'Return Value'] = rf + (beta_df['Beta Value'][i] * (rm - rf))

    with col2:
        st.markdown('Calculated Return using CAPM')
        st.dataframe(return_df,use_container_width=True)

except:
    st.write("Please select valid input")





