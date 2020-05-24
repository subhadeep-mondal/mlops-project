#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential


# In[6]:


model = Sequential()


# In[7]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))


# In[8]:


#model.summary()


# In[9]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[10]:


#model.summary()


# In[11]:


model.add(Flatten())


# In[12]:


#model.summary()


# In[13]:


model.add(Dense(units=128, activation='relu'))


# In[14]:


#model.summary()


# In[15]:


model.add(Dense(units=1, activation='sigmoid'))


# In[16]:


#model.summary()


# In[17]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[18]:


from keras.preprocessing.image import ImageDataGenerator


# In[19]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=[64,64],
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=[64,64],
        class_mode='binary')


# In[36]:


no_of_epocs = 2
history = model.fit(
            training_set,
            steps_per_epoch=100,
            epochs=no_of_epocs,
            validation_data=test_set,
            validation_steps=10)


# In[33]:


#accuracy = history.history['accuracy'][(no_of_epocs-1)] * 100
#accuracy


# In[34]:


#f=open("accuracy.txt",'w')
#f.write("%d" % int(history.history['accuracy'][(no_of_epocs-1)] * 100))
#f.close()


# In[54]:


loss, acc = model.evaluate(test_set, verbose=2)
accuracy = print("{:5.0f}".format(100*acc))


# In[ ]:





# In[ ]:




