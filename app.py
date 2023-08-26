import streamlit as st
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import yfinance as yf
import pickle
from utils import set_background

set_background("./imgs/background.png")

def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

company = 'AAPL'

#Put here the Model Path
MODEL_PATH = f'./models/{company}_model.h5'
LR_MODEL_PATH = f'./lr_models/{company}_lr_model.pkl'

header = st.container()
body = st.container()

header_2 = st.container()
body_2 = st.container()

def model_prediction(model, company):
    start = dt.datetime(2020,1,1)
    end = dt.datetime.now()

    if (company == "BTC" or company == "ETH") :
        data = yf.download(f"{company}-USD", start , end)
    else :
        data = yf.download(company, start , end)

    scaler = load_pkl(f"./scalers/{company}_scaler.pkl")

    new_df = data.filter(["Close"])
    last_60_days = new_df[-60:].values
    last_60_days = scaler.transform(last_60_days)

    x_pred_tomorrow = []
    x_pred_tomorrow.append(last_60_days)
    x_pred_tomorrow = np.array(x_pred_tomorrow)
    x_pred_tomorrow = np.reshape(x_pred_tomorrow, (x_pred_tomorrow.shape[0], x_pred_tomorrow.shape[1], 1))

    tomorrow_price = model.predict(x_pred_tomorrow)
    tomorrow_price = scaler.inverse_transform(tomorrow_price)
    tomorrow_price = "%.2f" % tomorrow_price[0][0]
    
    return tomorrow_price

def lr_model_prediction(model, stock) :
    start = dt.datetime(2020,1,1)
    end = dt.datetime.now()

    if (stock == "BTC" or stock == "ETH") :
        data = yf.download(f"{stock}-USD", start , end)
    else :
        data = yf.download(stock, start , end)

    ct = load_pkl(f"./lr_scalers/{stock}_lr_scaler.pkl")

    last_date = data.iloc[-1:]
    real_pred = ct.transform(last_date[["Open", "Volume"]])

    tomorrow_price = model.predict(real_pred)
    tomorrow_price = "%.2f" % tomorrow_price[0]
    
    return tomorrow_price

def stock_lstm_ver() :
    with header :
        st.title("Price Stock Prediction 💰")
        st.subheader("LSTM Machine Learning Algorithm 🌐")

    with body :
        company = st.selectbox("Stock or Crypto: ", options=["AAPL", "GOOG", "NFLX", "AMZN", "GS", "BTC", "ETH"], index=0)
        MODEL_PATH = f'./models/{company}_model.h5'

        model=''

        if model=='':
            model = tf.keras.models.load_model(
                (MODEL_PATH),
                custom_objects={'KerasLayer':hub.KerasLayer}
            )

        st.header("Actual vs Predicted Price Graphic 📊:")

        st.image(f"./graphics/{company}_graphic.png")

        _, col2, col3 = st.columns([0.3, 0.5, 0.9])

        col2.subheader("Check It-out ✅: ")

        if col3.button("Predict Price"):
            prediction = model_prediction(model, company)
            st.success(f"Price of the {company} Stock for Tomorrow: "  +  prediction + "$")    

def stock_lr_ver() :
    with header_2 :
        st.title("Price Stock Prediction 💰")
        st.subheader("Linear Regression Machine Learning Algorithm 🌐")

    with body_2 :
        stock = st.selectbox("Stock or Crypto: ", options=["AAPL", "GOOG", "NFLX", "AMZN", "GS", "BTC", "ETH"], index=0)
        LR_MODEL_PATH = f'./lr_models/{stock}_lr_model.pkl'

        lr_model='' 

        if lr_model=='':
            lr_model = load_pkl(LR_MODEL_PATH)

        st.header("Actual vs Predicted Price Graphic 📊:")

        st.image(f"./graphics/{stock}_lr_graphic.png")

        _, col2, col3 = st.columns([0.3, 0.5, 0.9])

        col2.subheader("Check It-out ✅:")

        if col3.button("Predict Price"):
            prediction = lr_model_prediction(lr_model, stock)
            st.success(f"Price of the {stock} Stock for Tomorrow: "  +  prediction + "$")  

page = st.sidebar.selectbox("Machine Learning Algorithm:", options=("LSTM", "Linear Regression"))

if page == "LSTM":
    stock_lstm_ver()
else :
    stock_lr_ver()




