import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

"""
побудувати модель регресії для датасету із ДЗ1 (ціна будинків). як лос використати loss = 10 * mse(x, y). 
Модель має включати лише 1 hidden layer. додати dropout регуляризацію до шару. зберегти модель
"""

data = pd.read_csv('kc_house_data.csv')
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

X = data.iloc[:, 3:15].values
y = data['price'].values

# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# creating keras model
model = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1),
    ]
)


# loss
def loss(y_true, y_pred):
    mse = keras.losses.mean_squared_error(y_true, y_pred)
    return 10 * mse


optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss=loss)

# train model
history = model.fit(X_train, y_train, epochs=30)

# evaluate model
mse = model.evaluate(X_test, y_test)
print(f'Test Loss: {mse}')

# saving model
model.save('house_price_regression_model.keras')
