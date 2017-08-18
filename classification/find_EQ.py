#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.io as sio
from astropy.time import Time
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.signal as sig


# read in data
H1dat = sio.loadmat('Data/' + 'H1_SeismicBLRMS.mat')
edat  = np.loadtxt('Data/'  + 'H1_earthquakes.txt')

#read in earthquake channels
cols   = [6,12,18,24,30,36,42,48]
vdat   = np.array(H1dat['data'][0])
vchans = np.array(H1dat['chans'][0])
for i in cols:
    add    = np.array(H1dat['data'][i])
    vdat   = np.vstack((vdat, add))
    vchans = np.append(vchans,H1dat['chans'][i])

#convert time to gps time                      
times   = '2017-03-01 00:00:00'
t       = Time(times,format='iso',scale='utc')
t_start = int(np.floor(t.gps/60)*60)
dur_in_days = 30
dur_in_minutes = dur_in_days*24*60
dur     = dur_in_minutes * 60
t_end   = t_start + dur
t       = np.arange(t_start,t_end, 60)
t0      = t[0]

# Find peaks using scipy CWT
widths  = np.arange(3,40)   # range of widths in minutes
peaks   = sig.find_peaks_cwt(vdat[0], widths, min_snr = 5)

if __debug__:
    print("This is something to do with peaks")
    print(peaks)

EQ_locations = np.array([])
for i in peaks:
    EQ_locations = np.append(EQ_locations, t[i])

if __debug__:
    print("What are EQ locations?")
    print(EQ_locations)

fig,axes  = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, chan in zip(axes, vdat, vchans):
    ax.scatter(t - t0, data ,edgecolor='', c='purple',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'),
                   rasterized=True)
    ax.set_yscale('log')
    ax.set_ylim(9, 1.1e4)
    ax.set_xlabel('Time (after GPS ' + str(t0) + ') [s]')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(EQ_locations)):
        ax.axvline(x=(EQ_locations[e] - t0),
                       color = 'orange', alpha=0.3)
fig.tight_layout()

if __debug__:
    print("Saving Figure...")

fig.savefig('Figures/EQ_peaks_indicated.pdf')

# can't have these hard coded path names; doesn't run for anyone else this way
#fig.savefig('/home/roxana.popescu/public_html/'+'EQ_peaks_indicated.png')
