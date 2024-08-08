# importing libraries
import streamlit as st
import datetime
import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
#import CAPM_functions
import numpy as np
import plotly.express as px

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

# setting page config
st.set_page_config(

        page_title="CAPM",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

st.title('Calculate Beta and Return for individual stock')

# getting input from user
col1, col2 = st.columns([1,1])
with col1:
    stock = st.selectbox("Choose a stock" , ('TSLA', 'AAPL','NFLX','MGM','MSFT','AMZN','NVDA','GOOGL'))
with col2:
    year = st.number_input("Number of Years",1,10)

# downloading data for SP500
end = datetime.date.today()
start = datetime.date(datetime.date.today().year - year, datetime.date.today().month, datetime.date.today().day)
SP500 = web.DataReader(['sp500'], 'fred', start, end)

# downloading data for the stock
stocks_df = yf.download(stock, start=start,end=end)
stocks_df = stocks_df[['Close']]
stocks_df.columns = [f'{stock}']
stocks_df.reset_index(inplace = True)
SP500.reset_index(inplace = True)
SP500.columns = ['Date','sp500']
stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
stocks_df['Date'] = stocks_df['Date'].apply(lambda x:str(x)[:10])
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
stocks_df = pd.merge(stocks_df, SP500, on = 'Date', how = 'inner')

# calculating daily return 
#stocks_daily_return = CAPM_functions.daily_return(stocks_df)
stocks_daily_return = daily_return(stocks_df)
rm = stocks_daily_return['sp500'].mean()*252

# calculate beta and alpha
#beta, alpha = CAPM_functions.calculate_beta(stocks_daily_return, stock)
beta, alpha = calculate_beta(stocks_daily_return, stock)
# risk free rate of return
rf = 2

# market potfolio return
rm = stocks_daily_return['sp500'].mean()*252

# calculate return
return_value = round(rf+(beta*(rm-rf)),2)

# showing results
st.markdown(f'### Beta : {beta}')
st.markdown(f'### Return  : {return_value}')
fig = px.scatter(stocks_daily_return, x = 'sp500', y = stock, title = stock)
fig.add_scatter(x = stocks_daily_return['sp500'], y = beta*stocks_daily_return['sp500'] + alpha,  line=dict(color="crimson"))
st.plotly_chart(fig, use_container_width=True)