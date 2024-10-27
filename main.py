from scripts import train, predict

if __name__ == "__main__":
    model = train.train_model()  # Huấn luyện mô hình và trả về mô hình đã huấn luyện
    predict.make_predictions(model=model)  # Truyền mô hình đã huấn luyện vào hàm dự đoán
