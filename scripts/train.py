import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils import load_data

def preprocess_data(data):
    lookback = 50  # Tăng số ngày đầu vào lên 300
    close_data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    train_data = []
    target_data = []
    for i in range(lookback, len(scaled_data)):
        train_data.append(scaled_data[i-lookback:i, 0])
        target_data.append(scaled_data[i, 0])

    return np.array(train_data), np.array(target_data), scaler


def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(300, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(300, return_sequences=True),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(200, return_sequences=True),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(150),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(1)
    ])
    
    # Thay đổi optimizer và loss function
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001), loss=tf.keras.losses.Huber())
    return model


def train_model():
    data = load_data.load_stock_data('BTC-USD')
    train_data, target_data, scaler = preprocess_data(data)
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

    model = build_lstm_model((train_data.shape[1], 1))
    
    # Tăng epoch và điều chỉnh batch size
    model.fit(train_data, target_data, epochs=100, batch_size=32, validation_split=0.2)
    
    model.save('model/stock_lstm_model.h5')
    print("Mô hình đã được huấn luyện và lưu thành công.")
    return model
