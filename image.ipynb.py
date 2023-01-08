#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


import os
import warnings
warnings.simplefilter('ignore')


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[6]:


DAEMON=os.listdir("C:/Users/STUDENT/Desktop/DATASET/DAEMON")


# In[7]:


JONSNOW=os.listdir("C:/Users/STUDENT/Desktop/DATASET/JON SNOW")


# In[8]:


ROBERTSTRAK=os.listdir("C:/Users/STUDENT/Desktop/DATASET/ROBERT STARK")


# In[9]:


limit =10
daemon_images=[None]*limit
j=0
for i in DAEMON:
    if(j<limit):
        daemon_images[j]=imread("C:/Users/STUDENT/Desktop/DATASET/DAEMON/"+i)
        j+=1
    else:
        break
        


# In[10]:


limit =10
jonsnow_images=[None]*limit
j=0
for i in JONSNOW:
    if(j<limit):
        jonsnow_images[j]=imread("C:/Users/STUDENT/Desktop/DATASET/JON SNOW/"+i)
        j+=1
    else:
        break
        


# In[11]:


limit =10
robertstark_images=[None]*limit
j=0
for i in ROBERTSTRAK:
    if(j<limit):
        robertstark_images[j]=imread("C:/Users/STUDENT/Desktop/DATASET/ROBERT STARK/"+i)
        j+=1
    else:
        break
        
    


# In[12]:


imshow(daemon_images[3])


# In[13]:


imshow(jonsnow_images[6])


# In[14]:


imshow(robertstark_images[8])


# In[15]:


daemon_gray=[None]*limit
j=0
for i in DAEMON:
    if(j<limit):
        daemon_images[j]=rgb2gray(daemon_images[j])
        j+=1
    else:
        break
        


# In[16]:


jonsnow_gray=[None]*limit
j=0
for i in JONSNOW:
    if(j<limit):
        jonsnow_images[j]=rgb2gray(jonsnow_images[j])
        j+=1
    else:
        break


# In[17]:


robertstark_gray=[None]*limit
j=0
for i in ROBERTSTRAK:
    if(j<limit):
        robertstark_images[j]=rgb2gray(robertstark_images[j])
        j+=1
    else:
        break


# In[18]:


imshow(daemon_images[3])


# In[19]:


imshow(jonsnow_images[6])


# In[20]:


imshow(robertstark_images[8])


# In[21]:


jonsnow_images[3].shape


# In[22]:


daemon_images[3].shape
  


# In[23]:


for i in range(10):
    k=daemon_images[i]
    daemon_images[i]=resize(k,(512,512))


# In[24]:


for i in range(10):
    a=jonsnow_images[i]
    jonsnow_images[i]=resize(a,(512,512))


# In[25]:


for i in range(10):
    b=robertstark_images[i]
    robertstark_images[i]=resize(b,(512,512))


# In[26]:


image_size_daemon_images=daemon_images[0].shape
image_size_jonsnow_images=jonsnow_images[0].shape
image_size_robertstark_images=robertstark_images[0].shape


# In[27]:


flatten_daemon_images=image_size_daemon_images[0]*image_size_daemon_images[1]
flatten_jonsnow_images=image_size_jonsnow_images[0]*image_size_jonsnow_images[1]
flatten_robertstark_images=image_size_robertstark_images[0]*image_size_robertstark_images[1]


# In[28]:


len_of_daemon_images=len(daemon_images)
len_of_jonsnow_images=len(jonsnow_images)
len_of_robertstark_images=len(robertstark_images)


# In[29]:


for i in range(len_of_daemon_images):
    daemon_images[i]=np.ndarray.flatten(daemon_images[i].reshape(flatten_daemon_images,1))


# In[30]:


for i in range(len_of_jonsnow_images):
    jonsnow_images[i]=np.ndarray.flatten(jonsnow_images[i].reshape(flatten_jonsnow_images,1))


# In[31]:


for i in range(len_of_robertstark_images):
    robertstark_images[i]=np.ndarray.flatten(robertstark_images[i].reshape(flatten_robertstark_images,1))


# In[32]:


daemon_images=np.dstack(daemon_images)
jonsnow_images=np.dstack(jonsnow_images)
robertstark_images=np.dstack(robertstark_images)


# In[33]:


daemon_images=np.rollaxis(daemon_images,axis=2,start=0)
jonsnow_images=np.rollaxis(jonsnow_images,axis=2,start=0)
robertstark_images=np.rollaxis(robertstark_images,axis=2,start=0)


# In[34]:


daemon_images=daemon_images.reshape(len_of_daemon_images,flatten_daemon_images)
jonsnow_images=jonsnow_images.reshape(len_of_jonsnow_images,flatten_jonsnow_images)
robertstark_images=robertstark_images.reshape(len_of_robertstark_images,flatten_robertstark_images)


# In[35]:


daemon_images.shape


# In[36]:


jonsnow_images.shape


# In[37]:


robertstark_images.shape


# In[38]:


daemon_data=pd.DataFrame(daemon_images)
jonsnow_data=pd.DataFrame(jonsnow_images)
robertstark_data=pd.DataFrame(robertstark_images)


# In[39]:


daemon_data


# In[40]:


jonsnow_data


# In[41]:


robertstark_data


# In[42]:


daemon_data["label"]="daemon"


# In[43]:


jonsnow_data["label"]="jonsnow"
robertstark_data["label"]="robertstark"


# In[44]:


daemon_data


# In[45]:


jonsnow_data


# In[46]:


robertstark_data


# In[47]:


actor_1=pd.concat([daemon_data,jonsnow_data])


# In[48]:


actor=pd.concat([actor_1,robertstark_data])


# In[49]:


actor


# In[50]:


from sklearn.utils import shuffle


# In[51]:


hollywood_indexed=shuffle(actor).reset_index()


# In[52]:


hollywood_indexed


# In[53]:


hollywood=hollywood_indexed.drop([0],axis=1)


# In[54]:


x=hollywood.values[:,:-1]


# In[55]:


y=hollywood.values[:,-1]


# In[56]:


x


# In[57]:


y


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[60]:


from sklearn import svm


# In[61]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# In[62]:


y_pred=clf.predict(x_test)


# In[63]:


y_pred


# In[64]:


for i in(np.random.randint(0,6,4)):
    predicted_images=(np.reshape(x_test[i],(512,512)).astype(np.float64))
    plt.title('predicted label:{0}'.format(y_pred[i]))
    plt.imshow(predicted_images,interpolation='nearest',cmap='gray')
    plt.show()


# In[65]:


from sklearn import metrics


# In[66]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[67]:


accuracy


# In[72]:


from sklearn.metrics import confusion_matrix


# In[73]:


confusion_matrix(y_test,y_pred)


# In[ ]:




