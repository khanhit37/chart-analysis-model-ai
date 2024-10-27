import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils import load_data

def make_predictions(model=None):
    if model is None:
        model = tf.keras.models.load_model('model/stock_lstm_model.h5')
    
    data = load_data.load_stock_data('BTC-USD')
    close_data = data['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    last_200_days = scaled_data[-200:].reshape(1, 200, 1)
    
    predictions = []
    for day in range(10):  
        predicted_price = model.predict(last_200_days)
        predictions.append(predicted_price[0, 0])
        last_200_days = np.append(last_200_days[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    extended_data = np.concatenate((close_data, predictions), axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(extended_data, color='blue', label='True and Predicted')
    plt.plot(range(len(close_data), len(extended_data)), predictions, color='orange', linestyle='--', marker='o', label='Prediction')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction with LSTM')
    plt.legend()
    plt.show()
