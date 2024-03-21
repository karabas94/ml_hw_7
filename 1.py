import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.keras
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

"""
побудувати модель багатокласової класифікації за допомогою keras
виконання має включати
    препроцесинг датасету
    розбиття трейн/тест
    побудова моделі керас
    callback на збереження найкращої моделі
    візуалізація процесу навчання за допомогою тензорборд
    EarlyStopping якщо зміна лосу на valid датасеті менша, ніж 1e-3
    тестування моделі на тестовому датасеті
    збереження кращої моделі
"""

data = pd.read_csv('WineQT.csv')

# first five row
print(f'First five row:\n {data.head()}')
print('\n')

# info
print(f'Info:\n{data.info()}')
print('\n')

# describe
print(f'Describe:\n{data.describe()}')
print('\n')

# count space in column
print(f'Count space in column:\n{data.isnull().sum()}')
print('\n')

# max in column
print(f'Max value of column:\n{data.max()}')
print('\n')

# min in column
print(f'Min value of column:\n{data.min()}')
print('\n')

# count of unique in column
print(f'Count of unique values in column:\n{data.nunique()}')
print('\n')

# check for duplicates
print(f'Number of duplicates: {data.duplicated().sum()}')
print('\n')

X = data.iloc[:, :-2].values
y = data['quality'].values

# splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_classes = 9

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = (X_train.shape[1],)

# creating keras model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ]
)
print(model.summary())

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

# callbacks
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', verbose=1, mode='min', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='min',
                               restore_best_weights=True)
# tensorboard = TensorBoard(log_dir='D:/logs', histogram_freq=1)
print("\n")

# visualization with mlflow
mlflow.keras.autolog()

# train model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[
    checkpoint,
    early_stopping,
    # tensorboard
])

# testing model with test data set
score = model.evaluate(X_test, y_test, verbose=1)
print(f'Test loss: {score[0]}')
print(f'Test Accuracy: {score[1]}')

# saving best model
model.save("final_model.keras")
