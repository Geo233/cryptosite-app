#from .date import extrage_date, salveaza_date_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

def main_prediction_function():
    crypto_data = pd.read_csv('bitcoin_prices_yahoo.csv')
    crypto_data = crypto_data.sort_values(by="Date")
    crypto_data['Close_diff'] = crypto_data['Close'].diff()
    crypto_data.dropna(subset=['Close_diff'], inplace=True)

    data = crypto_data[['Close_diff']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    def create_dataset(data, window_size=5):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    window_size = 5
    X, y = create_dataset(data_scaled, window_size)

    # LSTM Model
    model_lstm = Sequential([
        LSTM(50, input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.4),
        LSTM(50),
        Dropout(0.4),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # GRU Model
    model_gru = Sequential([
        GRU(50, input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.4),
        GRU(50),
        Dropout(0.4),
        Dense(1)
    ])
    model_gru.compile(optimizer='adam', loss='mean_squared_error')
    model_gru.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # Linear Regression
    X_train_linear = X.reshape(X.shape[0], -1)
    model_linear = Ridge().fit(X_train_linear, y)

    # ARIMA
    model_arima = ARIMA(y, order=(5, 1, 0))
    model_arima_fit = model_arima.fit()

    # Predict next day's price difference using all models
    last_window = data_scaled[-window_size:].reshape(1, window_size, 1)
    next_day_diff_lstm = model_lstm.predict(last_window)
    next_day_diff_gru = model_gru.predict(last_window)
    last_window_linear = last_window.reshape(1, -1)
    next_day_diff_linear = model_linear.predict(last_window_linear).reshape(-1, 1)
    next_day_diff_arima = model_arima_fit.forecast(steps=1)[0]

    # Combine the predictions
    combined_diff = (next_day_diff_lstm + next_day_diff_gru
                     + next_day_diff_linear + next_day_diff_arima) / 4
    combined_diff_original_scale = scaler.inverse_transform(combined_diff)

    # Calculate next day's close price
    last_day_close = crypto_data['Close'].iloc[-1]
    next_day_close = last_day_close + combined_diff_original_scale[0][0]

    print(f"Predicted price for the next day (combining both models): ${next_day_close:.2f}")
    print(combined_diff_original_scale[0][0])

    return (round(next_day_close, 2), round(combined_diff_original_scale[0][0], 2), round(last_day_close))
