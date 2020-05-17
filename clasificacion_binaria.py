
from keras.layers import Input, Flatten, Dense, Dropout, Reshape, PReLU,LeakyReLU,BatchNormalization,Activation
from keras.models import Model
import numpy as np
import keras
import keras.layers as layers
import keras.backend as K
import math
from keras.models import load_model,Model
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


traindata = pd.read_csv('titanic/train.csv')

testdata = pd.read_csv('titanic/test.csv')

def process_data(datos):
    xt = datos[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    xt = xt.fillna(xt.mean())

    xt.loc[:,'Sex'] = xt['Sex'].replace(['female','male'],[0,1]).values
    xt.loc[:,'Pclass'] = xt.loc[:,'Pclass'] - 1

    return xt

xt = process_data(traindata)

xtest = process_data(testdata)

yt = traindata["Survived"]
ytonehot = tf.keras.utils.to_categorical(yt, num_classes=2)
print(ytonehot)

arraytrainx=np.array(xt)
arraytrainy=np.array(yt)


Entradas = Input(shape=(6,))
x=Dense(200)(Entradas)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.05)(x)
#x=Dropout(0.15)(x)
x=Dense(50)(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.05)(x)
#x=Dropout(0.1)(x)
x=Dense(1)(x)
x = Activation('sigmoid')(x)

modelo = Model(inputs=Entradas, outputs=x)

Guardado = keras.callbacks.ModelCheckpoint('titanic.h5', monitor='val_acc', verbose=0, save_best_only=True,save_weights_only=False, mode='auto', period=1)

Adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.9)

modelo.compile(optimizer=Adam,loss='binary_crossentropy',metrics=['accuracy'])
history=modelo.fit(arraytrainx,arraytrainy ,epochs=400, batch_size=200,validation_split=0.2,callbacks=[Guardado],verbose=1)

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precision de Modelo')
plt.ylabel('Precision')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')


plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perdidas de Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')
plt.show()


