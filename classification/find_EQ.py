import numpy as np
import scipy.io as sio
from astropy.time import Time
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import signal

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

#convert time to gps time                      
times = '2017-03-01 00:00:00'
t = Time(times,format='iso',scale='utc')
t_start= int(np.floor(t.gps/60)*60)
dur_in_days= 30
dur_in_minutes = dur_in_days*24*60
dur = dur_in_minutes*60
t_end = t_start+dur
xvals = np.arange(t_start,t_end, 60)

widths = np.arange(35,40)
peaks = signal.find_peaks_cwt(vdat[0], widths)

print(peaks)

EQ_locations = np.array([])
for i in peaks:
    EQ_locations = np.append(EQ_locations, xvals[i])
print(EQ_locations)

fig,axes  = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, chan in zip(axes, vdat, vchans):
    ax.scatter(xvals, data ,edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlabel('GPS Time')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(EQ_locations)):
        ax.axvline(x=EQ_locations[e], color = 'r')
fig.tight_layout()
fig.savefig('/home/roxana.popescu/public_html/'+'EQ_peaks_indicated.png')
