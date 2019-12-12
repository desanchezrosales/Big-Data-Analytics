from PIL import Image
import glob 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import time
import h5py

start = time.time()

fname = '/fs/scratch/PAS1585/sanchezrosales1/train_labels/train_labels.csv'

df = pd.read_csv(fname)

short = False

if short:
    pixels = []
    label = []
    for i in range(100):
        pixels.append(np.asarray(Image.open( '/fs/scratch/PAS1585/sanchezrosales1/train/' + df['id'][i] + '.tif' )))
        label.append(df['label'][i])

else:
    pixels = []
    label = []
    for i in range(len(df)):
        pixels.append(np.array(Image.open( '/fs/scratch/PAS1585/sanchezrosales1/train/' + df['id'][i] + '.tif' )))
        label.append(df['label'][i])
        

hf = h5py.File('/fs/scratch/PAS1585/sanchezrosales1/training_data.h5', 'w')
hf.create_dataset('data', data=pixels)
hf.close()
        
end = time.time()

print(end - start)
