
# coding: utf-8

# In[2]:


import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# In[3]:


train_file = "fashionmnist/fashion-mnist_train.csv"
test_file = "fashionmnist/fashion-mnist_test.csv"

img_rows, img_cols, img_channel = 28, 28, 1
input_shape = (img_rows, img_cols, img_channel)

num_classes = 10

train_label_dic = {0: "T-shirt/top",
                  1: "Trouser",
                  2: "Pullover",
                  3: "Dress",
                  4: "Coat",
                  5: "Sandal",
                  6: "Shirt",
                  7: "Sneaker",
                  8: "Bag",
                  9: "Ankle boot"}


# In[4]:


# Read input from csv file
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
X_train = np.array(train.iloc[:,1:])
Y_train = to_categorical(np.array(train.iloc[:,0]))
X_test = (np.array(test.iloc[:,1:]))
Y_test = to_categorical(np.array(test.iloc[:,0]))

# Split training data into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=2)

# Get number of examples in each set
N_train = X_train.shape[0]
N_val = X_val.shape[0]
N_test = X_test.shape[0]

# Reshape to 3D images
X_train = X_train.reshape(N_train, img_rows, img_cols, img_channel)
X_val = X_val.reshape(N_val, img_rows, img_cols, img_channel)
X_test = X_test.reshape(N_test, img_rows, img_cols, img_channel)

# Normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255


# In[5]:


index = 2
plt.imshow(X_train[index, :, :, 0])
print("Label = ", train_label_dic[list(Y_train[index]).index(1)])


# In[6]:


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                 kernel_initializer='he_normal', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
          
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
          
model.add(Dense(num_classes, activation='softmax'))


# In[7]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[8]:


model.summary()


# In[ ]:


plot_model(model, show_shapes=True, to_file='model.png')
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# In[ ]:


model.fit(X_train, Y_train, batch_size=256, epochs=20, verbose=1, validation_data=(X_val, Y_val))


# In[ ]:


score = model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])

