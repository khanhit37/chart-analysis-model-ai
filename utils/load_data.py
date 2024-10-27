import yfinance as yf

def load_stock_data(ticker, period="2y"):
    # Mặc định tải dữ liệu 2 năm gần nhất nếu không chỉ định
    stock_data = yf.download(ticker, period=period, interval="1d")
    return stock_data
