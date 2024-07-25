import tkinter as tk
from tkinter import ttk
import ccxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
import threading
import time
import ta
import requests
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from joblib import dump, load

class BitcoinApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bitcoin Price and Indicator Predictions")
        
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        self.side_frame = tk.Frame(root, width=200)
        self.side_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.exchange = ccxt.binance()
        
        self.fig, (self.ax, self.ax2, self.ax3) = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.create_side_panel()
        self.load_or_train_models()
        self.update_data()

    def create_side_panel(self):
        self.sentiment_label = tk.Label(self.side_frame, text="Sentiment:", font=("Helvetica", 12))
        self.sentiment_label.pack(pady=10)
        
        self.pressure_label = tk.Label(self.side_frame, text="Pressure:", font=("Helvetica", 12))
        self.pressure_label.pack(pady=10)
        
        self.volume_label = tk.Label(self.side_frame, text="Volume:", font=("Helvetica", 12))
        self.volume_label.pack(pady=10)

    def fetch_data(self):
        ohlcv = self.exchange.fetch_ohlcv('ETH/USDT', timeframe='30m', limit=500)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data

    def calculate_indicators(self, data):
        data['SMA_50'] = ta.trend.sma_indicator(data['close'], window=50)
        data['EMA_50'] = ta.trend.ema_indicator(data['close'], window=50)
        data['RSI'] = ta.momentum.rsi(data['close'], window=14)
        data['MACD'] = ta.trend.macd(data['close'])
        data['MACD_signal'] = ta.trend.macd_signal(data['close'])
        data['MACD_hist'] = ta.trend.macd_diff(data['close'])
        data['ADX'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14)
        data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
        data['CCI'] = ta.trend.cci(data['high'], data['low'], data['close'], window=14)
        data['OBV'] = ta.volume.on_balance_volume(data['close'], data['volume'])
        data['MFI'] = ta.volume.money_flow_index(data['high'], data['low'], data['close'], data['volume'], window=14)
        stochastic = ta.momentum.stoch(data['high'], data['low'], data['close'], window=14, smooth_window=3)
        data['Stochastic_K'] = stochastic
        data['Stochastic_D'] = stochastic.shift(1)
        return data

    def plot_candlestick(self, data):
        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        for idx, row in data.iterrows():
            color = 'green' if row['close'] >= row['open'] else 'red'
            self.ax.plot([row['timestamp'], row['timestamp']], [row['low'], row['high']], color='black')
            self.ax.plot([row['timestamp'], row['timestamp']], [row['open'], row['close']], color=color, linewidth=5)
        
        self.ax.plot(data['timestamp'], data['SMA_50'], label='SMA 50', color='blue')
        self.ax.plot(data['timestamp'], data['EMA_50'], label='EMA 50', color='cyan')
        self.ax.plot(data['timestamp'], data['MACD'], label='MACD', color='magenta')
        self.ax.plot(data['timestamp'], data['MACD_signal'], label='MACD Signal', color='yellow')
        
        self.ax2.plot(data['timestamp'], data['RSI'], label='RSI', color='purple')
        self.ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
        self.ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
        
        self.ax3.plot(data['timestamp'], data['ADX'], label='ADX', color='blue')
        
        self.ax.legend()
        self.ax2.legend()
        self.ax3.legend()
        
        self.fig.autofmt_xdate()
        self.canvas.draw()

    def update_side_panel(self, data):
        sentiment = self.get_sentiment()
        pressure = self.get_pressure(data)
        volume = data['volume'].iloc[-1]
        
        self.sentiment_label.config(text=f"Sentiment: {sentiment}")
        self.pressure_label.config(text=f"Pressure: {pressure}")
        self.volume_label.config(text=f"Volume: {volume:.2f}")

    def get_sentiment(self):
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url)
        sentiment_data = response.json()
        sentiment_value = sentiment_data["data"][0]["value"]
        sentiment_classification = sentiment_data["data"][0]["value_classification"]
        return f"{sentiment_classification} ({sentiment_value})"

    def get_pressure(self, data):
        latest_close = data['close'].iloc[-1]
        latest_open = data['open'].iloc[-1]
        pressure = (latest_close - latest_open) / latest_open * 100
        return f"{pressure:.2f}%"

    def prepare_features(self, data):
        features = data[['open', 'high', 'low', 'close', 'volume', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'ADX', 'ATR', 'CCI', 'OBV', 'MFI', 'Stochastic_K', 'Stochastic_D']].copy()
        features = features.fillna(0)  # Replace NaNs with zero for simplicity
        return features

    def create_lstm_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(50))
        model.add(Dropout(0.3))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(3))  # Predicting close price, RSI, and ADX
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def create_dnn_model(self, input_shape):
        model = Sequential()
        model.add(Dense(128, input_dim=input_shape[1], activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3))  # Predicting close price, RSI, and ADX
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def load_or_train_models(self):
        try:
            self.model_lstm = load_model('lstm_model.h5')
            self.model_dnn = load_model('dnn_model.h5')
            self.model_lr = load('lr_model.joblib')
            self.model_rf = load('rf_model.joblib')
        except:
            self.train_and_save_models()

    def train_and_save_models(self):
        data = self.fetch_data()
        data = self.calculate_indicators(data)
        features = self.prepare_features(data)
        X = features[:-5]
        y = data[['close', 'RSI', 'ADX']][5:]  # Predicting close price, RSI, and ADX
        
        y = y.fillna(method='ffill').fillna(method='bfill')

        # Linear Regression
        self.model_lr = LinearRegression()
        self.model_lr.fit(X, y)
        dump(self.model_lr, 'lr_model.joblib')
        
        # Random Forest
        self.model_rf = RandomForestRegressor(n_estimators=100)
        self.model_rf.fit(X, y)
        dump(self.model_rf, 'rf_model.joblib')
        
        # LSTM
        look_back = 60
        X_lstm, y_lstm = [], []
        for i in range(look_back, len(features)):
            X_lstm.append(features[i-look_back:i].values)
            y_lstm.append(data[['close', 'RSI', 'ADX']].iloc[i].values)
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], X_lstm.shape[2]))
        
        self.model_lstm = self.create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
        self.model_lstm.fit(X_lstm, y_lstm, epochs=20, batch_size=64, verbose=1)
        self.model_lstm.save('lstm_model.h5')

        # DNN
        self.model_dnn = self.create_dnn_model(X.shape)
        self.model_dnn.fit(X, y, epochs=20, batch_size=64, verbose=1)
        self.model_dnn.save('dnn_model.h5')

    def predict_next_bars(self, data):
        features = self.prepare_features(data)
        X = features[-60:]  # Last 60 data points for prediction

        # Linear Regression
        forecast_lr = self.model_lr.predict(X[-5:])
        
        # Random Forest
        forecast_rf = self.model_rf.predict(X[-5:])
        
        # LSTM
        X_lstm = features[-60:].values.reshape((1, 60, features.shape[1]))
        forecast_lstm = self.model_lstm.predict(X_lstm)
        forecast_lstm = np.tile(forecast_lstm, (5, 1))  # Assurez-vous que forecast_lstm a 5 prédictions

        # DNN
        forecast_dnn = self.model_dnn.predict(X[-5:])

        # Vérifiez que toutes les colonnes de prévision ont la même longueur
        assert len(forecast_lr) == 5
        assert len(forecast_rf) == 5
        assert forecast_lstm.shape[0] == 5
        assert len(forecast_dnn) == 5

        future_index = pd.date_range(start=data['timestamp'].iloc[-1], periods=6, freq='30min')[1:]
        forecast_df = pd.DataFrame({
            'timestamp': future_index,
            'LR_close': forecast_lr[:, 0], 'LR_RSI': forecast_lr[:, 1], 'LR_ADX': forecast_lr[:, 2],
            'RF_close': forecast_rf[:, 0], 'RF_RSI': forecast_rf[:, 1], 'RF_ADX': forecast_rf[:, 2],
            'LSTM_close': forecast_lstm[:, 0], 'LSTM_RSI': forecast_lstm[:, 1], 'LSTM_ADX': forecast_lstm[:, 2],
            'DNN_close': forecast_dnn[:, 0], 'DNN_RSI': forecast_dnn[:, 1], 'DNN_ADX': forecast_dnn[:, 2]
        })
        return forecast_df

    def plot_predictions(self, forecast_df):
        self.ax.plot(forecast_df['timestamp'], forecast_df['LR_close'], label='LR Close Forecast', color='cyan', linestyle='--')
        self.ax.plot(forecast_df['timestamp'], forecast_df['RF_close'], label='RF Close Forecast', color='magenta', linestyle='--')
        self.ax.plot(forecast_df['timestamp'], forecast_df['LSTM_close'], label='LSTM Close Forecast', color='orange', linestyle='--')
        self.ax.plot(forecast_df['timestamp'], forecast_df['DNN_close'], label='DNN Close Forecast', color='green', linestyle='--')
        
        self.ax2.plot(forecast_df['timestamp'], forecast_df['LR_RSI'], label='LR RSI Forecast', color='cyan', linestyle='--')
        self.ax2.plot(forecast_df['timestamp'], forecast_df['RF_RSI'], label='RF RSI Forecast', color='magenta', linestyle='--')
        self.ax2.plot(forecast_df['timestamp'], forecast_df['LSTM_RSI'], label='LSTM RSI Forecast', color='orange', linestyle='--')
        self.ax2.plot(forecast_df['timestamp'], forecast_df['DNN_RSI'], label='DNN RSI Forecast', color='green', linestyle='--')
        
        self.ax3.plot(forecast_df['timestamp'], forecast_df['LR_ADX'], label='LR ADX Forecast', color='cyan', linestyle='--')
        self.ax3.plot(forecast_df['timestamp'], forecast_df['RF_ADX'], label='RF ADX Forecast', color='magenta', linestyle='--')
        self.ax3.plot(forecast_df['timestamp'], forecast_df['LSTM_ADX'], label='LSTM ADX Forecast', color='orange', linestyle='--')
        self.ax3.plot(forecast_df['timestamp'], forecast_df['DNN_ADX'], label='DNN ADX Forecast', color='green', linestyle='--')
        
        self.ax.legend()
        self.ax2.legend()
        self.ax3.legend()

    def update_data(self):
        data = self.fetch_data()
        data = self.calculate_indicators(data)
        self.plot_candlestick(data)
        self.update_side_panel(data)
        
        forecast_df = self.predict_next_bars(data)
        self.plot_predictions(forecast_df)
        
        self.root.after(60000, self.update_data)  # Update every 60 seconds

if __name__ == "__main__":
    root = tk.Tk()
    app = BitcoinApp(root)
    root.mainloop()
