#!/usr/bin/env python
# coding: utf-8

# In[63]:


#Import packages
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from face_detector import FaceDetector


# In[64]:


#Load images
img=cv2.imread('Images/surprise.png')


# In[65]:


#Resize the image
img=cv2.resize(img,(600,600))


# In[66]:


#Sclae image
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[67]:


#Visualize the image
plt.imshow(img)


# In[68]:


# Load the classifiers 
face_cascade = cv2.CascadeClassifier('Files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Files/haarcascade_eye.xml')


# In[69]:


#Detect face and eyes (Find the face and eyes in the image and draw a rectangle)
faces=face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)


# In[70]:


#Visualize the image
plt.imshow(img)


# In[73]:


#Predictions(emotions,race,age)
predections=DeepFace.analyze(img)


# In[74]:


#print the predections
print(predections)


# In[75]:


# print the dominant emotion
predections['dominant_emotion']
print(predections['dominant_emotion'])


# In[76]:


# print the age
predections['age']
print(predections['age'])


# In[77]:


# print the gender
predections['gender']
print(predections['gender'])


# In[78]:


# print the race
predections['dominant_race']
print(predections['dominant_race'])


# In[79]:


#Add text to the image
cv2.putText(img=img, text=predections['dominant_emotion'], org=(50, 50), 
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, 
            color=(0, 0, 255),thickness=3);


# In[80]:


#Visualize the image
plt.imshow(img)


# In[140]:





# In[ ]:




