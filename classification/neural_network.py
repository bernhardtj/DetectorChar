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
H1dat = sio.loadmat('Data/' + 'H1_SeismicBLRMS.mat')
edat = np.loadtxt('Data/H1_earthquakes.txt')

#read in earthquake channels
cols = [6,12,18,24,30,36,42,48]
vdat = np.array(H1dat['data'][0])
vchans = np.array(H1dat['chans'][0])
for i in cols:
    add = np.array(H1dat['data'][i])
    vdat = np.vstack((vdat, add))
    vchans = np.append(vchans,H1dat['chans'][i])

#shift the data
t_shift = 10 #how many minutes to shift the data by
for i in cols:
    add = np.array(H1dat['data'][i])
    for j in range(1, t_shift+1):        
        add_shift = add[j:]
        #print(np.shape(add_shift))
        add_values = np.zeros((j,1))
        add_shift = np.append(add_shift, add_values)
        #print(np.shape(add_shift))
        vdat = np.vstack((vdat, add_shift))
        chan = 'Time_Shift_' + str(j) + '_Min_EQ_Band_' + str(i)
        vchans = np.append(vchans, chan)
vdat = vdat[:,:43200-t_shift]

#create list of earthquake times                                
#convert time to gps time                      
times = '2017-03-01 00:00:00'
t = Time(times,format='iso',scale='utc')
t_start= int(np.floor(t.gps/60)*60)
dur_in_days= 30
dur_in_minutes = dur_in_days*24*60
dur = dur_in_minutes*60
t_end = t_start+dur
#use peak ground motion to determine which earthquakes are bigger  
row, col = np.shape(edat)
gdat = np.array([])
for i in range(row):
    point = edat[i][20]
    gdat = np.append(gdat,point)
gdat = gdat.T
glq = np.percentile(gdat,65)
print(glq)
print(min(gdat))
#use only earthquakes with signifigant ground motion                                                                                           
row, col = np.shape(edat)
etime = np.array([])
for i in range(row):
    if (edat[i][20] >= glq):
        point = edat[i][5]
        etime = np.append(etime,point)
#use only earthqaukes that occur in March 2017 
col = len(etime)
etime_march = np.array([])
for i in range(col):
    if ((etime[i] >= t_start) and (etime[i] <= t_end)):
        point = etime[i]
        etime_march = np.append(etime_march,point)

#assign X
X = vdat.T
#assign Y to 1 or 0 depending on wheter there is an earthquake
xvals = np.arange(t_start,t_end-t_shift*60, 60)
Y = np.array([])
print(len(xvals))
for x in xvals:
    xlen = len(Y)
    for j in etime_march:
        if (x <= j+30*60 and x >= j-30*60):
            Y = np.append(Y,1)
    xlen2 = len(Y)
    if xlen == xlen2:
        Y  = np.append(Y,0)

#neural network
optimizer = optimizers.Adam(lr = 1e-5)
model = Sequential()
model.add(Dense(89, input_shape = (89,), activation = 'elu'))
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
Y_pred3 = Y_pred3.astype(int)

'''
colors = np.array(['r', 'b', 'm', 'g'])
labels = Y.T
labels = labels.astype(int)
xvals = np.arange(t_start,t_end,60)
fig,axes  = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, chan in zip(axes, vdat, vchans):
    ax.scatter(xvals, data,c=colors[y_pred],edgecolor='',
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
    fig.savefig('/home/roxana.popescu/public_html/'+'NeuralNetworkComparison2.png')
except FileNotFoundError:
    fig.savefig('Figures/NeuralNetworkComparison2.png')
'''
