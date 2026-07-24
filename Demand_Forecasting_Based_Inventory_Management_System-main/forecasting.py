import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from math import sqrt
import tensorflow as tf
import logging

#harddiski dolduran tensorflow loglarıymış bu şekilde kapatılıyormuş
tf.get_logger().setLevel(logging.ERROR)

def pinball_loss(q=0.90):

    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e), axis=-1)
    return loss

def create_sequences(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)

def train_forecast_model(series, model_type="GRU", window=6, horizon=6, epochs=50, loss_type="mse", quantile=0.90):

    if len(series) < window + 3:
        return None, None, {}, None

    split = int(len(series) * 0.8)
    train, val = series[:split], series[split:]

    if len(val) < window:
        return None, None, {}, None

    scaler = MinMaxScaler()
    scaler.fit(train.values.reshape(-1, 1))

    train_scaled = scaler.transform(train.values.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val.values.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_scaled, window)
    X_val, y_val = create_sequences(val_scaled, window)

    if len(X_train) == 0 or len(X_val) == 0:
        return None, None, {}, None

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    # --- MODEL MİMARİSİ ---
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(32, input_shape=(window, 1)))
    else:
        model.add(GRU(32, input_shape=(window, 1)))

    model.add(Dropout(0.2))
    model.add(Dense(1))

    if loss_type == "pinball":
        model.compile(optimizer="adam", loss=pinball_loss(q=quantile))
        monitor_metric = 'val_loss'
    else:
        model.compile(optimizer="adam", loss="mse")
        monitor_metric = 'val_loss'

    early_stop = EarlyStopping(monitor=monitor_metric, patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=epochs, verbose=0, callbacks=[early_stop])

#validasyon predictleri
    val_preds = scaler.inverse_transform(
        model.predict(X_val, verbose=0)
    ).flatten()
    y_val_real = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_val_real, val_preds)
    rmse = sqrt(mean_squared_error(y_val_real, val_preds))
    mape = np.mean(np.abs((y_val_real - val_preds) / (y_val_real + 1e-5))) * 100


    errors = y_val_real - val_preds
    pinball_score = np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))

    # geleceği tahminle
    full_data_scaled = scaler.transform(series.values.reshape(-1, 1)).flatten()
    last_window = full_data_scaled[-window:]

    forecast_values = []
    current = last_window
    for _ in range(horizon):
        pred = model.predict(current.reshape(1, window, 1), verbose=0)[0][0]
        forecast_values.append(pred)
        current = np.append(current[1:], pred)

    forecast_values = scaler.inverse_transform(
        np.array(forecast_values).reshape(-1, 1)
    ).flatten()

    ma_forecast = [series[-window:].mean()] * horizon

    metrics = {
        "MAPE": round(mape, 2),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "PinballLoss": round(pinball_score, 4)
    }

    return forecast_values, ma_forecast, metrics, val_preds