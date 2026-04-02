import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Загрузка CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot для categorical_crossentropy и MSE
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Бинарные метки (0-4 vs 5-9) для binary_crossentropy
y_train_bin = (y_train >= 5).astype("float32")
y_test_bin = (y_test >= 5).astype("float32")


# Архитектура CNN для categorical/MSE (10 классов)
def build_cnn_10classes():
    model = models.Sequential(
        [
            layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)
            ),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


# Архитектура CNN для бинарной классификации (выход 1 нейрон, сигмоида)
def build_cnn_binary():
    model = models.Sequential(
        [
            layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)
            ),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


# Функция обучения и оценки
def train_evaluate(
    model, x_train, y_train, x_test, y_test, loss, epochs=30, batch_size=64
):
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        verbose=1,
    )
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return test_acc, history


print("Обучение с categorical_crossentropy...")
model_cat = build_cnn_10classes()
acc_cat, hist_cat = train_evaluate(
    model_cat,
    x_train,
    y_train_cat,
    x_test,
    y_test_cat,
    "categorical_crossentropy",
    epochs=30,
)

print("\nОбучение с binary_crossentropy (бинарная задача 0-4 vs 5-9)...")
model_bin = build_cnn_binary()
acc_bin, hist_bin = train_evaluate(
    model_bin,
    x_train,
    y_train_bin,
    x_test,
    y_test_bin,
    "binary_crossentropy",
    epochs=30,
)

print("\nОбучение с mean_squared_error...")
model_mse = build_cnn_10classes()
acc_mse, hist_mse = train_evaluate(
    model_mse, x_train, y_train_cat, x_test, y_test_cat, "mean_squared_error", epochs=30
)

# Вывод результатов
print("\n========== РЕЗУЛЬТАТЫ ==========")
print(f"categorical_crossentropy  →  точность: {acc_cat*100:.2f}%")
print(f"binary_crossentropy       →  точность (бинарная): {acc_bin*100:.2f}%")
print(f"mean_squared_error        →  точность: {acc_mse*100:.2f}%")

# Графики
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_cat.history["accuracy"], label="categorical")
plt.plot(hist_bin.history["accuracy"], label="binary")
plt.plot(hist_mse.history["accuracy"], label="MSE")
plt.title("Training accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_cat.history["val_accuracy"], label="categorical")
plt.plot(hist_bin.history["val_accuracy"], label="binary")
plt.plot(hist_mse.history["val_accuracy"], label="MSE")
plt.title("Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
