# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:32:05 2019

@author: danis
"""

import numpy as np
#
# Used to implement the multi-dimensional counter we need in the performance class
from collections import defaultdict
def autovivify(levels=1, final=dict):
    return (defaultdict(final) if levels < 2 else
            defaultdict(lambda: autovivify(levels-1, final)))
def getPerformance(network,images,labels_cat,labels):
#
# Get the overall performance for the test sample
    loss, acc = network.evaluate(images,labels_cat)
#
# Get the individual predictions for each sample in the test set
    predictions = network.predict(images)
#
# Get the max probabilites for each rows
    probs = np.max(predictions, axis = 1)
#
# Get the predicted classes for each row
    classes = np.argmax(predictions, axis = 1)
#
# Now loop over the first twenty samples and compare truth to prediction
#print("Label\t Pred\t Prob")
#for label,cl,pr in zip(smear_labels[:20],classes[:20],probs[:20]):
#    print(label,'\t',cl,'\t',round(pr,3))
#
# Get confustion matrix
    cf = autovivify(2,int)
    for label,cl in zip(labels,classes):
        cf[label][cl] += 1
#
    return loss,acc,cf
###############################################################################
    
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#
# Change the folling to False to run on the full data
# NOTE: Keep true when running interactively!!
short = False
if short:
    train_images = train_images[:7000,:]
    train_labels = train_labels[:7000]
    test_images = test_images[:3000,:]
    test_labels = test_labels[:3000]
#
print("Train info",train_images.shape, train_labels.shape)
print("Test info",test_images.shape, test_labels.shape)
train_images = train_images.reshape((train_images.shape[0],28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((test_images.shape[0],28*28))
test_images = test_images.astype('float32')/255
from keras.utils import to_categorical

train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)


###############################################################################

from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
#
# Make sure the shape of the input is correct (the last ",1" is the number of "channels"=1 for grayscale)
train_images = train_images.reshape((train_images.shape[0],28,28,1))
test_images = test_images.reshape((test_images.shape[0],28,28,1))
#
cnn_network = models.Sequential()
#
# First convolutional layer
cnn_network.add(layers.Conv2D(30,(5,5),activation='relu',input_shape=(28,28,1)))
# Pool
cnn_network.add(layers.MaxPooling2D((2,2)))
#
# Second convolutional layer
cnn_network.add(layers.Conv2D(25,(5,5),activation='relu'))
# Pool
cnn_network.add(layers.MaxPooling2D((2,2)))
#
# Connect to a dense output layer - just like an FCN
cnn_network.add(layers.Flatten())
cnn_network.add(layers.Dense(64,activation='relu'))
cnn_network.add(layers.Dense(10,activation='softmax'))
#
# Compile
cnn_network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#

patienceCount = 10
callbacks = [EarlyStopping(monitor='val_loss', patience=patienceCount),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]


# Fit/save/print summary
history = cnn_network.fit(train_images,train_labels_cat,epochs=50,batch_size=256,validation_data=(test_images,test_labels_cat))
cnn_network.save('fully_trained_model_cnn.h5')
#print(cnn_network.summary())
#
# Get the overall performance for the test sample
test_loss, test_acc = cnn_network.evaluate(test_images,test_labels_cat)
print("Test sample loss: ",test_loss, "; Test sample accuracy: ",test_acc)

loss,acc,cf = getPerformance(cnn_network,test_images,test_labels_cat,test_labels)

print("Test confusion matrix")
for trueClass in range(10):
    print("True: ",trueClass,end="")
    for predClass in range(10):
        print("\t",cf[trueClass][predClass],end="")
    print()
print()
