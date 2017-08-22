#!/usr/bin/env python

'''
Reads in data from mat file
Creates a neural network using keras
Plots data with prediction labels in a graph 
'''

from __future__ import division
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers 
import numpy as np
import scipy.io as sio
from astropy.time import Time
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from bilinearHelper import *
# Hush tensorflow warnings about AVX instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_plot_style()

#np.random.seed(7)

seconds_per_day = 24*60*60

# read in data
EQ_data  = sio.loadmat('Data/EQ_info.mat')
vdat     = EQ_data['odat']
vchans   = EQ_data['vchans']
EQ_times = EQ_data['EQ_times'].T
X        = EQ_data['X']
Npoints, Ncols  = X.shape

Y        = EQ_data['EQ_labels'].T
Y        = np.squeeze(Y)       # remove the singleton dim
t_shift  =  len(Y) - Npoints    # find out how big the time window is
Y        = Y[t_shift:]         # drop the first t_shift pts
t        = EQ_data['t'].T      # transpose me because I don;t fit



# print info about data
if __debug__:
    print('Shape of BLRMS is ' + str(X.shape))
    print('Shape of Y (labels) is ' + str(Y.shape))

# neural network
isize = Ncols
optimizer = optimizers.Adam(lr = 1e-5)
dropout = 0.05

model = Sequential()
model.add(Dense(isize, input_shape = (isize,), activation = 'elu'))
model.add(Dropout(.1))
#model.add(Dense(isize, input_shape = (isize,), activation = 'elu'))
#model.add(Dropout(.1))

model.add(Dense(2**6, activation = 'elu'))
model.add(Dropout(dropout))
model.add(Dense(2**3, activation = 'elu'))
model.add(Dropout(dropout))

model.add(Dense(1, activation = 'sigmoid'))

# model.output_shape
model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
convBLRMS = model.fit(X, Y,
              epochs           = 7,
              batch_size       = 128,
              validation_split = 0.5,
              verbose          = 1)

#score = model.evaluate(X,Y)
#print(score)
model.save('EQ-Classification-Model.h5')
model.summary()

# prediction values 
Y_pred  = model.predict(X)
Y_pred2 = Y_pred.T
Y_pred3 = np.array([])
for i in Y_pred2:
    Y_pred3 = np.append(Y_pred3, i)
Y_pred3 = Y_pred3.astype(int)

# Plot of data points 
colors = np.array(['r', 'b', 'm', 'g'])
labels = Y.T
labels = labels.astype(int)
Nsensors = 9                 # 3 seismometers, with 3 axes each

t0 = t[0]
tday = (t - t[0])/seconds_per_day
 
fig,axes  = plt.subplots(len(vdat[0:Nsensors]),
                             figsize = (40,4*len(vdat[0:Nsensors])),
                             sharex  = True)
for ax, data, chan in zip(axes, vdat[0:Nsensors], vchans):

    ax.scatter(tday[:,0], data,
                   c = 'purple', edgecolor='', s=3,
                   label=r'$\mathrm{%s}$' % chan.replace('_','\_'),
                   rasterized=True, alpha = 0.7)
    ax.set_yscale('log')
    for k in range(len(labels)):
        if labels[k] > 0:
            ax.axvline(x = tday[t_shift+k,0],
                           lw=2, linestyle='--',
                           color='green', alpha=0.5)
    
    ax.set_ylim(9, 1.1e4)
    ax.set_xlim(0, 30)
    ax.set_xticks(range(30))
    ax.set_xlabel('Time (after GPS ' + str(t0) + ') [days]')
    ax.grid(True, which='both')
    ax.legend()
    for e in EQ_times:
        e = (e - t0)/seconds_per_day
        ax.axvline(x = e,
                       lw=4, color='orange', alpha=0.3)

fig.tight_layout()

print("Saving plot...")
fig.savefig('Figures/ConvNet-Comparison.pdf')


plot_training_progress(convBLRMS, plotDir = 'Figures') 
