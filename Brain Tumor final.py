
import os
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 


encoder = OneHotEncoder()
encoder.fit([[0], [1]]) 

# 0 - Tumor
# 1 - Normal

data = []
paths = []
result = []

for r, d, f in os.walk(r'D:\\Education\\New Dataset\\yes 121'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())


paths = []
for r, d, f in os.walk(r"D:\\Education\\New Dataset\\no33"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())       


data = np.array(data)
data.shape

result = np.array(result)
len(result)
result = result.reshape(235,2)
print(result)

x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.1, shuffle=True, random_state=0)


from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeUniform

import tensorflow as tf

model = Sequential()
activation = 'relu'
model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation = activation, padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation = activation, padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation = activation, padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))


model.add(Dense(2 ,activation ='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
#print(model.summary())


y_train.shape

history = model.fit(x_train, y_train, epochs = 11, batch_size = 40, verbose = 1,validation_data = (x_test, y_test))

def names(number):
    if number==0:
        return 'it has  a Tumor'
        a==1
    else:
        return 'there are no tumor'
        a==0

import matplotlib.pyplot as plt
img = Image.open(r"D:\\Education\\New Dataset\\pred\\pred32.jpg")
plt.xlabel('Sample Image')
plt.imshow(img)
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]

print(str(res[0][0]*100) + '% Confidence ' + names(classification))       


print(model.evaluate(x_test, y_test))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()


acc[10]*100 #accuracy %