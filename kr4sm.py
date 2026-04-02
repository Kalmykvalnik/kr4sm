import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import urllib.request
import zipfile

# 1. ЗАГРУЗКА ДАННЫХ В ЛОКАЛЬНУЮ ПАПКУ (без использования .keras)
print("Загрузка данных...")

# Папка для данных рядом со скриптом
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "climate_data")
os.makedirs(data_dir, exist_ok=True)

zip_path = os.path.join(data_dir, "jena_climate_2009_2016.csv.zip")
csv_path = os.path.join(data_dir, "jena_climate_2009_2016.csv")

# Если CSV уже есть, не качаем заново
if not os.path.exists(csv_path):
    # Скачиваем вручную через urllib
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    print(f"Скачивание {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print("Распаковка...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    print("Готово.")
else:
    print("CSV файл уже существует, пропускаем загрузку.")

df = pd.read_csv(csv_path)

print("Первые 5 строк данных:")
print(df.head())
print(f"\nРазмер данных: {df.shape}")

temperature = df["T (degC)"].values.reshape(-1, 1)

# 2. ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_scaled = scaler.fit_transform(temperature)


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


SEQUENCE_LENGTH = 30
X, y = create_sequences(temperature_scaled, SEQUENCE_LENGTH)

train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"\nФорма X_train: {X_train.shape}")
print(f"Форма X_test: {X_test.shape}")

# 3. МОДЕЛЬ
model = tf.keras.Sequential(
    [
        tf.keras.layers.SimpleRNN(
            50, activation="tanh", input_shape=(SEQUENCE_LENGTH, 1)
        ),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print("\nАрхитектура модели:")
model.summary()

# 4. ОБУЧЕНИЕ
print("\nНачало обучения модели...")
history = model.fit(
    X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=1
)

# 5. ОЦЕНКА
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nОценка на тестовых данных:")
print(f"MSE: {test_loss:.4f}")
print(f"MAE: {test_mae:.4f}")

y_pred = model.predict(X_test)

y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = scaler.inverse_transform(y_pred)

# ВИЗУАЛИЗАЦИЯ
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test_original[:500], label="Actual Temperature", color="blue")
plt.plot(y_pred_original[:500], label="Predicted Temperature", color="red", alpha=0.7)
plt.title("Actual vs Predicted Temperature")
plt.xlabel("Time Step")
plt.ylabel("Temperature (°C)")
plt.legend()

plt.tight_layout()
plt.show()

print("\nСравнение первых 10 предсказаний:")
for i in range(10):
    print(
        f"Шаг {i+1}: Реальное = {y_test_original[i][0]:.2f}°C, Предсказанное = {y_pred_original[i][0]:.2f}°C"
    )
