import time
import h5py
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from keras import models
from keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping

start = time.time()

#load images
hf = h5py.File('/fs/scratch/PAS1585/sanchezrosales1/Train_data_100k.h5', 'r')
hf.keys()
pics = hf.get('data')

#load label
hf2 = h5py.File('/fs/scratch/PAS1585/sanchezrosales1/Train_labels_100k.h5', 'r')
hf2.keys()
label = hf2.get('data')

#prepare data
images = pics[0:]/255
labels = to_categorical(label[0:])

    
cnn_network = models.Sequential()

# First convolutional layer
cnn_network.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(96,96,3)))
cnn_network.add(layers.MaxPooling2D((2,2)))

# Second convolutional layer
cnn_network.add(layers.Conv2D(32,(3,3),activation='relu'))
cnn_network.add(layers.MaxPooling2D((2,2)))

# Connect to a dense output layer - just like an FCN
cnn_network.add(layers.Flatten())
cnn_network.add(layers.Dense(75,activation='relu'))
cnn_network.add(layers.Dense(2,activation='softmax'))

# Compile
cnn_network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

   
history = cnn_network.fit(images,labels,epochs=4,batch_size=64,verbose = 1)
print('Train accuracy:',history.history['acc'])
print('Test Loss:' ,  history.history['loss'])
        
cnn_network.save('/fs/scratch/PAS1585/sanchezrosales1/Trained_Models/best_model_100k.h5')

end = time.time()
print(end - start)
