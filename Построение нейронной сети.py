#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import plotly.express as px
import tensorflow as tf
import sklearn

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from tensorflow import keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LeakyReLU, Flatten, Dropout, BatchNormalization, Activation, LSTM
from pandas import read_excel, DataFrame, Series
from scikeras.wrappers import KerasClassifier, KerasRegressor
# from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor --depricated
from tensorflow.keras.models import Sequential
from numpy.random import seed
from scipy import stats
import warnings



import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

print(tf.__version__)


# In[56]:


df_matrix = pd.read_csv('Datasets/df_clean.csv')
df.head(2)


# In[69]:


pip install scikeras


# In[57]:


df_matrix.shape


# In[58]:


target_matrix = df_matrix['Соотношение матрица-наполнитель']
train_matrix = df_matrix.drop(['Соотношение матрица-наполнитель', 'Unnamed: 0'], axis=1)


# In[59]:


target_matrix.head(2)


# In[60]:


train_matrix.head(2)


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(train_matrix, target_matrix, test_size = 0.3, random_state = 17)


# In[62]:


# нормализуем входные данные и преобразуем в np.array
x_train_n = tf.keras.layers.Normalization(axis =-1)
x_train_n.adapt(np.array(x_train))


# In[71]:


model = Sequential(x_train_n)

model.add(Dense(128))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(32))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(32))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(1))
model.add(Activation(activation='elu'))


# In[72]:


model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False),loss='mean_absolute_error')


# In[74]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(x_train, y_train,\n    batch_size = 64,\n    epochs=40,\n    verbose=1,\n    validation_split = 0.2\n    )')


# In[75]:


model.summary()


# In[80]:


#Функция для построения графика потерь модели на тренировочной и тестовой выборках
def model_loss_plot(model_history):
    plt.figure(figsize=(10, 5))
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('График потерь модели', size=12)
    plt.ylabel('Средняя абсолютная ошибка', size=12)
    plt.xlabel('Эпоха', size=12)
    plt.legend(['loss', 'val_loss'], loc='best')
    plt.show()

#Функция для построения графика оригинального и предсказанного значения у
def actual_and_predicted_plot(original_y, predicted_y):    
    plt.figure(figsize=(10,5))
    plt.title('Тестовые и прогнозные значения', size=12)
    plt.plot(original_y, color='blue', label = 'Тестовые значения')
    plt.plot(predicted_y, color='violet', label = 'Прогнозные значения')
    plt.legend(loc='best')
    plt.show()

#Функция для построения точечного графика оригинального и предсказанного значения у   
def actual_and_predicted_scatter(original_y, predicted_y):
    plt.figure(figsize=(10,5))
    plt.title('Рассеяние тестовых и прогнозных значений', size=15)
    plt.scatter(original_y, predicted_y)
    plt.xlabel('Тестовые значения', size=12)
    plt.ylabel('Прогнозные значения', size=12)
    plt.show()


# In[81]:


model_loss_plot(history)


# In[82]:


predicted = model.predict(np.array((x_test)))
original = y_test.values

actual_and_predicted_plot(original, predicted)


# In[83]:


actual_and_predicted_scatter(original, predicted)


# In[84]:


print(f'Model MAE: {model.evaluate(x_test, y_test, verbose=1)}')


# In[85]:


print(f'MAE среднего значения: {np.mean(np.abs(y_test-np.mean(y_test)))}')


# In[86]:


model2 = Sequential(x_train_n)

model2.add(Dense(128))
model2.add(BatchNormalization())
model2.add(LeakyReLU())
model2.add(Dense(128, activation='selu'))
model2.add(BatchNormalization())
model2.add(Dense(64, activation='selu'))
model2.add(BatchNormalization())
model2.add(Dense(32, activation='selu'))
model2.add(BatchNormalization())
model2.add(LeakyReLU())
model2.add(Dense(16, activation='selu'))
model2.add(BatchNormalization())
model2.add(Dense(1))
model2.add(Activation('selu'))


# In[87]:


early_model2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')


# In[88]:


model2.compile(optimizer=tf.optimizers.SGD(learning_rate=0.02, momentum=0.5), loss='mean_absolute_error')


# In[89]:


get_ipython().run_cell_magic('time', '', 'history2 = model2.fit(\n    x_train,\n    y_train,\n    batch_size = 64,\n    epochs=100,\n    verbose=1,\n    validation_split = 0.2,\n    callbacks = [early_model2]\n    )')


# In[90]:


model2.summary()


# In[91]:


model_loss_plot(history2)


# In[92]:


predicted2 = model2.predict(np.array((x_test)))
original2 = y_test.values

actual_and_predicted_plot(original2, predicted2)


# In[93]:


actual_and_predicted_scatter(original2, predicted2)


# In[94]:


print(f'Model MAE: {model2.evaluate(x_test, y_test)}')


# In[95]:


print(f'MAE среднего значения: {np.mean(np.abs(y_test-np.mean(y_test)))}')


# In[ ]:




