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
hf = h5py.File('/fs/scratch/PAS1585/sanchezrosales1/data.h5', 'r')
hf.keys()
pics = hf.get('data')

#load label
fname = '/fs/scratch/PAS1585/sanchezrosales1/train_labels/train_labels.csv'
labels_df = pd.read_csv(fname)

#prepare data
images = pics[0:]/255
labels = to_categorical(labels_df['label'][0:])

#prepare folds
kfolds = 5
skf = KFold(n_splits=kfolds, random_state=42, shuffle=True)

#prepare indices for shuffling
indexes = range(len(images))

count_2 = 0

#########################################################################################################################
#prepare fitter
def run_fitter(l1,k1,l2,k2,hn):
    
    cnn_network = models.Sequential()

    # First convolutional layer
    cnn_network.add(layers.Conv2D(l1,(k1,k1),activation='relu',input_shape=(96,96,3)))
    cnn_network.add(layers.MaxPooling2D((2,2)))
    
    # Second convolutional layer
    cnn_network.add(layers.Conv2D(l2,(k2,k2),activation='relu'))
    cnn_network.add(layers.MaxPooling2D((2,2)))

    # Connect to a dense output layer - just like an FCN
    cnn_network.add(layers.Flatten())
    cnn_network.add(layers.Dense(hn,activation='relu'))
    cnn_network.add(layers.Dense(2,activation='softmax'))

    # Compile
    cnn_network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    #set up early stopping
    patienceCount = 10
    callbacks = [EarlyStopping(monitor='val_loss', patience=patienceCount)]
    
    avg_acc_test = 0.0
    avg_acc_train = 0.0
    avg_loss_test = 0.0
    avg_loss_train = 0.0
    avg_epochs = 0.0
    numSplits = 0.0
    count = 0
    
    for train_index, test_index in skf.split(indexes):
        print('Training')
        X_train = images[train_index]
        y_train = labels[train_index]
        X_test = images[test_index]
        y_test = labels[test_index]
        
        history = cnn_network.fit(X_train,y_train,epochs=50,batch_size=64,verbose = 0, callbacks = callbacks, validation_data=(X_test,y_test))

        training_vals_acc = history.history['acc']
        training_vals_loss = history.history['loss']
        valid_vals_acc = history.history['val_acc']
        valid_vals_loss = history.history['val_loss']

        avg_epochs += len(training_vals_acc) - patienceCount

        avg_acc_train += training_vals_acc[-patienceCount]
        avg_loss_train += training_vals_loss[-patienceCount]

        avg_acc_test += valid_vals_acc[-patienceCount]
        avg_loss_test += valid_vals_loss[-patienceCount]

        numSplits += 1.0
        
        cnn_network.save('/fs/scratch/PAS1585/sanchezrosales1/Trained_Models/fully_trained_model_' + str(count_2) + '_' + str(count) + '.h5')
        count += 1        

    avg_acc_test /= (numSplits)
    avg_acc_train /= (numSplits)
    
    avg_loss_test /= (numSplits)
    avg_loss_train /= (numSplits)
    
    avg_epochs /= (numSplits)
    
    return avg_acc_test,avg_acc_train,avg_loss_test,avg_loss_train,avg_epochs

#########################################################################################################################
# run the fitter
start = time.time()

avg_acc_train_list = []
avg_loss_train_list = []

avg_acc_test_list = []
avg_loss_test_list = []

avg_epochs_list =[]

for nodes in [10,50,100,500]:
    
    avg_acc_test,avg_acc_train,avg_loss_test,avg_loss_train,avg_epochs = run_fitter(32,3,32,3,nodes)
    
    count_2 += 1

    avg_acc_train_list.append(avg_acc_train)
    avg_loss_train_list.append(avg_loss_train)

    avg_acc_test_list.append(avg_acc_test)
    avg_loss_test_list.append(avg_loss_test)
    
    avg_epochs_list.append(avg_epochs)

#########################################################################################################################
#prepare plots

def enable_plotly_in_cell():
  import IPython
  from plotly.offline import init_notebook_mode
#
# OLD (google colab)
#  display(IPython.core.display.HTML('''
#        <script src="/static/components/requirejs/require.js"></script>
#  '''))
#  init_notebook_mode(connected=False)
#
# New (OSC) [thanks to Stephen Gant for this!]
  init_notebook_mode(connected=True)

from plotly.offline import iplot
import plotly.graph_objs as go
import numpy as np

enable_plotly_in_cell()
#
# Costs
data1 = go.Scatter(
    x=np.array([10,50,100,500]),
    y=avg_acc_test_list,
    mode='lines',
    name="Avg Test Accuracy"
)

data1_2 = go.Scatter(
    x=np.array([10,50,100,500]),
    y=avg_acc_train_list,
    mode='lines',
    name="Avg Test Accuracy"
)
iplot(dict(data=[data1,data1_2]))

data2 = go.Scatter(
    x=np.array([10,50,100,500]),
    y=avg_loss_test_list,
    mode='lines',
    name="Avg Test Loss"
)
data2_1 = go.Scatter(
    x=np.array([10,50,100,500]),
    y=avg_loss_train_list,
    mode='lines',
    name="Avg Test Loss"
)

iplot(dict(data=[data2,data2_1]))

data3 = go.Scatter(
    x=np.array([10,50,100,500]),
    y=avg_epochs_list,
    mode='lines',
    name="Average N Epochs"
)
iplot(dict(data=[data3]))

end = time.time()
print(end - start)
