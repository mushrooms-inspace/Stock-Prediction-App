import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go

# Define START and TODAY variables
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# CSS styles
st.markdown(
    """
    <style>
        .title {
            color: #008080;
            text-align: center;
            padding: 10px;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 30px;
        }
        
        .main-content {
            text-align: center;
            margin-top: 20px;
        }

        .footer {
            text-align: center;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Stock Prediction App</div>', unsafe_allow_html=True)

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME' , 'GC=F')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data with enhanced styling
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open",
                             line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close",
                             line=dict(color='firebrick', width=2)))
    fig.update_layout(title_text='Time Series Data with Rangeslider',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      xaxis_rangeslider_visible=True,
                      template='plotly_white')
    st.plotly_chart(fig)

plot_raw_data()

# Define plot_plotly function with enhanced styling
def plot_plotly(model, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast',
                             line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None,
                             mode='lines', line=dict(color='rgba(0,176,246,0.2)', width=1), name='Lower Bound'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty',
                             mode='lines', line=dict(color='rgba(0,176,246,0.2)', width=1), name='Upper Bound'))
    fig.add_trace(go.Scatter(x=model.history['ds'], y=model.history['y'], name='Actual',
                             mode='lines', line=dict(color='yellow', width=2)))
    
    fig.update_layout(title_text='Forecast Plot',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      template='plotly_white')
    return fig

# Predict forecast with Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

st.markdown('<div class="footer">Made with ❤️</div>', unsafe_allow_html=True)
