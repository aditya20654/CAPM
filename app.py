# inmporting libraries
import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_datareader.data as web
#import CAPM_functions

import plotly.express as px
import numpy as np

#function to plot interactive plotly chart
def interactive_plot(df):
    fig= px.line()
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'],y=df[i],name=i)
    fig.update_layout(width=450, margin=dict(l=20,r=20,t=50,b=20),legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1))
    return fig

#function to normalize the prices based on the initial price
def normalize(df2):
    df=df2.copy()
    for i in df.columns[1:]:
        df[i]=df[i]/df[i][0] # to compare price of stock to its initial listed price    
    return df

#function to return daily return
def daily_return(df2):
    df_daily_return=df2.copy()
    for i in df_daily_return.columns[1:]:
        for j in range(1,len(df_daily_return)):
            df_daily_return[i][j]=((df2[i][j]-df2[i][j-1])/df2[i][j-1])*100
        df_daily_return[i][0]=0
    return df_daily_return

#function to calculate beta
def calculate_beta(stocks_daily_return,stock):
    rm=stocks_daily_return['sp500'].mean()*252 # expected yearly return of sp500
    
    b,a=np.polyfit(stocks_daily_return['sp500'],stocks_daily_return[stock],1)
    return b,a


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
        st.plotly_chart(interactive_plot(stock_df))
    with col2:
        st.markdown("Normalized price of all the stocks")
        st.plotly_chart(interactive_plot(normalize(stock_df)))
    #print(stock_df)
    stocks_daily_return=(daily_return(stock_df))

    beta={}
    alpha={}

    for i in stocks_daily_return.columns:
        if i!='Date' and i!='sp500':
            b,a=calculate_beta(stocks_daily_return,i)

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

    with col1:
        stock_ = st.selectbox("Choose a stock" , ('TSLA', 'AAPL','NFLX','MGM','MSFT','AMZN','NVDA','GOOGL'))
    with col2:
         year_ = st.number_input("Number of Years",1,10)
    beta, alpha = calculate_beta(stocks_daily_return, stock_)
    return_value = round(rf+(beta*(rm-rf)),2)
    st.markdown(f'### Beta : {beta}')
    st.markdown(f'### Return  : {return_value}')
    fig = px.scatter(stocks_daily_return, x = 'sp500', y = stock_, title = stock_)
    fig.add_scatter(x = stocks_daily_return['sp500'], y = beta*stocks_daily_return['sp500'] + alpha,  line=dict(color="crimson"))
    st.plotly_chart(fig, use_container_width=True)
except:
    st.write("Please select valid input")





