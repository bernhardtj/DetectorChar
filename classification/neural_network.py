'''
Reads in data from mat file
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

# read in data
EQ_data  = sio.loadmat('Data/EQ_info.mat')
vdat     = EQ_data['vdat']
vchans   = EQ_data['vchans']
EQ_times = EQ_data['EQ_times']
X        = EQ_data['X']
points, size  = np.shape(X)
Y        = EQ_data['EQ_labels']
Y        = Y.reshape(43200,)
t        = EQ_data['t']

# print info about data
print('size is ' + str(size))
print('Shape of X is ' + str(np.shape(X)))
print('Shape of Y is ' + str(np.shape(Y)))

# neural network
optimizer = optimizers.Adam(lr = 1e-5)
model = Sequential()
model.add(Dense(size, input_shape = (size,), activation = 'elu'))
model.add(Dropout(.1))
model.add(Dense(9, activation = 'elu'))
model.add(Dropout(.1))
model.add(Dense(1, activation = 'softmax'))

# model.output_shape
model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
model.fit(X, Y,
              epochs = 10,
              batch_size = 256,
              validation_split=0.1,
              verbose = 1)

#score = model.evaluate(X,Y)
#print(score)

model.summary()

# prediction values 
Y_pred = model.predict(X)
Y_pred2 = Y_pred.T
Y_pred3 = np.array([])
for i in Y_pred2:
    Y_pred3 = np.append(Y_pred3, i)
Y_pred3 = Y_pred3.astype(int)

# Plot of data points 
colors = np.array(['r', 'b', 'm', 'g'])
labels = Y.T
labels = labels.astype(int)
num  = size

fig,axes  = plt.subplots(len(vdat[0:num]), figsize=(40,4*len(vdat[0:num])))
for ax, data, chan in zip(axes, vdat[0:num], vchans):
    ax.scatter(t, data,c=colors[Y_pred3],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlabel('GPS Time')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(EQ_times)):
        print e
        #ax.axvline(x = EQ_times[e])

fig.tight_layout()

print("Saving plot...")
fig.savefig('Figures/NeuralNetworkComparison3.pdf')

try:
    fig.savefig('/home/roxana.popescu/public_html/'+'NeuralNetworkComparison3.png')
except:
    fig.savefig('Figures/NeuralNetworkComparison3.png')
    print('  ')
    
