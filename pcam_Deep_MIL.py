#!/usr/bin/env python
# coding: utf-8

# In[3]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt



# In[4]:

import keras
import tensorflow as tf
from tqdm import tqdm
from keras.utils import HDF5Matrix,multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D,Activation,BatchNormalization,LeakyReLU,GlobalAveragePooling2D
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from keras.regularizers import l2



# In[5]:




x_train = HDF5Matrix('/media/sdc/Xiankun/Histopathology/brenchmark_dataset/train/camelyonpatch_level_2_split_train_x.h5','x')
y_train = HDF5Matrix('/media/sdc/Xiankun/Histopathology/brenchmark_dataset/train_label/camelyonpatch_level_2_split_train_y.h5','y')

#x_validation = HDF5Matrix('/media/sdc/Xiankun/Histopathology/brenchmark_dataset/validation/camelyonpatch_level_2_split_valid_x.h5','x')
#y_validation = HDF5Matrix('/media/sdc/Xiankun/Histopathology/brenchmark_dataset/validation_label/camelyonpatch_level_2_split_valid_y.h5','y')

y_train = np.squeeze(y_train)
#y_validation = np.squeeze(y_validation)

#x_train, x_validation, y_train, y_validation = train_test_split(X_train, Y_train, test_size=0.1)





#datagen = ImageDataGenerator(preprocessing_function=lambda x: x/255., validation_split = 0.1)


# In[6]:





class noisyand(keras.layers.Layer):
    def __init__(self, num_classes, a = 20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1,a)
        super(noisyand,self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(name = "b",shape = (1,input_shape[-1]), initializer = "uniform",trainable = True)
        super(noisyand,self).build(input_shape)

    def call(self,x):
        mean = tf.reduce_mean(x, axis = [1,2])
        return (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


# In[7]:


def define_model(input_shape= (96,96,3), num_classes=1):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     padding = 'same',
                     input_shape=input_shape))
    
    model.add(Conv2D(64, (3, 3), padding = 'same', activation='relu'))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))  
    model.add(Conv2D(128, (1, 1), activation='relu'))
    
    model.add(noisyand(num_classes+1))    
    model.add(Dense(num_classes, activation='sigmoid'))
    
    return model


# In[8]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
		


# In[10]:


model = define_model()

model = multi_gpu_model(model, 2)

model.summary()


# In[11]:


model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer= keras.optimizers.RMSprop(),
                  metrics=['accuracy'])


# In[12]:


epoch=20
history = model.fit(x_train, y_train, batch_size=128,
        epochs=epoch,
        verbose=1, validation_split=0.1)


# In[ ]:


acc=history.history['acc']
epochs_=range(0,epoch)
plt.plot(epochs_,acc,label='training accuracy')

acc_val=history.history['val_acc']
plt.scatter(epochs_,acc_val,label="validation accuracy")
plt.ylim([0.6,1.0])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot of Model')

plt.legend()

plt.savefig('Accuracy Plot of Model')


# In[ ]:


acc=history.history['loss']
epochs_=range(0,epoch)
plt.plot(epochs_,acc,label='training loss')

acc_val=history.history['val_loss']
plt.scatter(epochs_,acc_val,label="validation loss")
plt.ylim([0,0.7])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot of Model')

plt.legend()

plt.savefig('Loss Plot of Model.png')


# In[ ]:


clf=model
y_pred_proba = clf.predict_proba(x_test)
y_pred = clf.predict_classes(x_test)

fpr,tpr,_= roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

plt.savefig('Receiver operating characteristic example.png')


# In[ ]:




