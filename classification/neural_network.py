'''
Reads in March 2017 earthquake channels and earthquake data
Shifts the earthquake channels by time
Makes a list of march earthquakes with peak ground motion greater than 65th percentile
Creates list that labels each point as near an earthquake or not
Creates a neural network using keras
Plots data with prediction labels in a graph 
'''

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

np.random.seed(7)

#read in data
EQ_data = sio.loadmat('Data/EQ_info.mat')
vdat = EQ_data['vdat']
vchans = EQ_data['vchans']
EQ_locations = EQ_data['EQ_locations']
X = EQ_data['X']
print(np.shape(X))
Y = EQ_data['Y']
Y = Y.T
print(np.shape(Y))
t = EQ_data['t']

#print info about data
size, points = np.shape(vdat)
num = size
print(' ')
print(size)
print(' ')

#neural network
optimizer = optimizers.Adam(lr = 1e-5)
model = Sequential()
model.add(Dense(size, input_shape = (size,), activation = 'elu'))
model.add(Dropout(.1))
model.add(Dense(9, activation = 'elu'))
model.add(Dropout(.1))
model.add(Dense(1, activation = 'softmax'))
#model.output_shape
model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.fit(X,Y, epochs = 10, batch_size = 256, verbose = 1)
score = model.evaluate(X,Y)
print(score)
model.summary()

#prediction values 
Y_pred = model.predict(X)
Y_pred2 = Y_pred.T
Y_pred2 = Y_pred2.astype(int)
Y_pred3 = np.array([])
for i in Y_pred2:
    Y_pred3 = np.append(Y_pred3, i)
Y_pred3 = np.astype(int)

#Plot of data points 
colors = np.array(['r', 'b', 'm', 'g'])
labels = Y.T
labels = labels.astype(int)
fig,axes  = plt.subplots(len(vdat[0:num]), figsize=(40,4*len(vdat[0:num])))
for ax, data, chan in zip(axes, vdat[0:num], vchans):
    ax.scatter(t, data,c=colors[Y_pred3],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlabel('GPS Time')
    ax.grid(True, which='both')
    ax.legend()
    #for e in range(len(etime_march)):
    #    ax.axvline(x=etime_march[e])
fig.tight_layout()
try:
    fig.savefig('/home/roxana.popescu/public_html/'+'NeuralNetworkComparison3.png')
except FileNotFoundError:
    fig.savefig('Figures/NeuralNetworkComparison2.png')

