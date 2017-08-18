from __future__ import division

import numpy as np
import scipy.io as sio
from astropy.time import Time
import collections

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as sig

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

#plt.style.use('seaborn-dark')
mpl.rcParams.update({
    'axes.grid': True,
    'axes.titlesize': 'medium',
    'font.family': 'serif',
    'font.size': 12,
    'grid.color': 'w',
    'grid.linestyle': '-',
    'grid.alpha': 0.5,
    'grid.linewidth': 1.5,
    'legend.borderpad': 0.2,
    'legend.fancybox': True,
    'legend.fontsize': 13,
    'legend.framealpha': 0.7,
    'legend.handletextpad': 0.1,
    'legend.labelspacing': 0.2,
    'legend.loc': 'best',
    'lines.linewidth': 1.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'text.usetex': False,
    'text.latex.preamble': r'\usepackage{txfonts}'
    })

mpl.rc("savefig", dpi=100)

# read in data
H1dat = sio.loadmat('Data/' + 'H1_SeismicBLRMS.mat')
edat  = np.loadtxt('Data/H1_earthquakes.txt')

#read in earthquake channels
cols   = [6,12,18,24,30,36,42,48]
vdat   = np.array(H1dat['data'][0])
vchans = np.array(H1dat['chans'][0])
for i in cols:
    add    = np.array(H1dat['data'][i])
    vdat   = np.vstack((vdat, add))
    vchans = np.append(vchans,H1dat['chans'][i])

#shift the data
t_shift = 0 #how many minutes to shift the data by
if t_shift > 0:
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
size, points = np.shape(vdat)
print(points)    

#convert time to gps time                      
times   = '2017-03-01 00:00:00'
t       = Time(times,format='iso',scale='utc')
t_start = int(np.floor(t.gps/60)*60)
dur_in_days = 30
dur_in_minutes = dur_in_days*24*60
dur     = dur_in_minutes * 60
t_end   = t_start + dur
t       = np.arange(t_start,t_end-t_shift*60, 60)
seconds_per_day = 24*60*60

# Find peaks using scipy CWT


if __debug__:
    print("This is something to do with peaks")
   # print(peaks)


# find peaks in all three z channel directions
widths  = np.arange(5, 140)   # range of widths in minutes
min_snr = 8
noise_perc = 15
peaks1 = sig.find_peaks_cwt(vdat[2], widths,
                                min_snr = min_snr, noise_perc=noise_perc)
peaks2 = sig.find_peaks_cwt(vdat[5], widths,
                                min_snr = min_snr, noise_perc=noise_perc)
peaks3 = sig.find_peaks_cwt(vdat[8], widths,
                                min_snr = min_snr, noise_perc=noise_perc)
peak_list = np.array([])

# use logical AND operation here instead
# needs to allow for slightly different arrival times at the different stations
for i in peaks1:
    for j in peaks2:
        for k in peaks3:
            if (abs(i-j) <= 2*60 and abs(i-k) <= 2*60):
                avg = (i+j+k)/3
                peak_list = np.append(peak_list, avg)
EQ_locations = np.array([])
for i in peak_list:
    EQ_locations = np.append(EQ_locations, t[int(i)])

#assign X
X = vdat.T
print(np.shape(X))
#assign Y to 1 or 0 depending on whether there is an earthquake
Y = np.array([])
for i in t:
    xlen = len(Y)
    for j in EQ_locations:
        if (j-5*60 <= i <= j+5*60):
            Y = np.append(Y,1)
            break
    xlen2 = len(Y)
    if xlen == xlen2:
        Y  = np.append(Y,0)
print(len(Y))
print(collections.Counter(Y))

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
Y_pred3 = Y_pred3.astype(int)

#Plot earthquakes determined by peaks
fig,axes  = plt.subplots(nrows=len(vdat), figsize=(40,4*len(vdat)),
                             sharex=True)
for ax, data, chan in zip(axes, vdat, vchans):
    ax.scatter((t - t[0])/seconds_per_day, data ,
                   edgecolor='', c='purple', s=3,
                   label=r'$\mathrm{%s}$' % chan.replace('_','\_'),
                   rasterized=True)
    ax.set_yscale('log')
    ax.set_ylim(9, 1.1e4)
    ax.set_ylabel('RMS Velocity [nm/s]') 
    ax.set_xlabel('Time (after GPS ' + str(t[0]) + ') [days]')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(EQ_locations)):
        ax.axvline(x=(EQ_locations[e] - t[0])/seconds_per_day,
                       color = 'blue', alpha=0.3, linewidth=3)

plt.xlim((0, (t[-1]-t[0])/seconds_per_day))
plt.title('Seismic BLRMS')
fig.tight_layout()

try:
    fig.savefig('/home/.popescu/public_html/'+'EQ_peaks_indicated.png')
except: 
    fig.savefig('Figures/EQ_peaks_indicated.pdf')
