#!/usr/bin/env python
# coding: utf-8

# In[16]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np


# In[17]:


#import dataset
from keras.datasets import mnist
(train_img,train_lab),(test_img, test_lab)= mnist.load_data()
print(test_lab[:30])


# In[18]:


#normalizing the dataset
train_img=train_img.reshape(60000,28,28,1)
test_img=test_img.reshape(10000,28,28,1)
train_img=keras.utils.normalize(train_img,axis=1)
test_img=keras.utils.normalize(test_img,axis=1)


# In[19]:


#Build the model
model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(3,3))

model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(3,3))

model.add(Flatten())
model.add(Dense(300,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[20]:


#make a summary for the model
model.summary()


# In[21]:


#Compiling the model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])


# In[22]:


#fitting the model
model.fit(train_img,train_lab,epochs=10)


# In[23]:


#evaluate the model
print(model.evaluate(test_img,test_lab))


# In[24]:


#predecting the first 10 test images
pred = model.predict(test_img[:10])
print(pred)
p = np.argmax(pred,axis=1)
print(p)
print(test_lab[:10])


# In[25]:


#visualizing prediction
for i in range(5):
    plt.imshow(test_img[i].reshape((28, 28)), cmap='binary')
    plt.title("Original:{},Predicted:{}".format(test_lab[i], p[i]))
    plt.axis("Off")
    plt.figure()


# In[ ]:




