#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


#Import Dataset
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# In[34]:


print(Y_train[:30])


# In[ ]:


#Normalise Dataset
X_train = keras.utils.normalize(X_train, axis=1)
X_test = keras.utils.normalize(X_test, axis =1)


# In[4]:


#Display the dataset shape - the traning dataset
print('X_train shape: ',X_train.shape)
print('Y_train shape: ',Y_train.shape)


# In[6]:


#Display the dataset shape - the test dataset
print('X_test shape: ',X_test.shape)
print('Y_test shape: ',Y_test.shape)


# In[9]:


#visualise the data -sample image
plt.imshow(X_train[400])
plt.show()
print(Y_train[400])


# In[10]:


#Build the model
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation="softmax"))


# In[11]:


#make a summary for the model
model.summary()


# In[13]:


#Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


# In[48]:


#Fit the Model
model.fit(X_train, Y_train, epochs=10)


# In[46]:


#Evaluate the model
history=model.evaluate(X_test, Y_test)
print(history)


# In[47]:


#Predicte the 10 first test images
prediction = model.predict(X_test[:10])
print(prediction)


# In[20]:


p=np.argmax(prediction, axis=1)
print(p)
print(Y_test[:10])


# In[22]:


#Visualizing prediction
for i in range(3):
  plt.imshow(X_test[i], cmap='binary')
  plt.title("Original: {}, Predicted: {}".format(Y_test[i], p[i]))
  plt.axis("Off")
  plt.figure()


# In[ ]:




