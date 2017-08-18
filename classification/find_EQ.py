#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy.io as sio
from astropy.time import Time
import collections

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy.signal as sig

plt.style.use('seaborn-dark')
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
seconds_per_day = 24*60*60

# Find peaks using scipy CWT
widths  = np.arange(5, 40)   # range of widths in minutes
peaks   = sig.find_peaks_cwt(vdat[2], widths,
                                 min_snr = 7,
                                 noise_perc=20)

if __debug__:
    print("This is something to do with peaks")
    print(peaks)

EQ_locations = np.array([])
for i in peaks:
    EQ_locations = np.append(EQ_locations, t[i])

if __debug__:
    print("What are EQ locations?")
    print(EQ_locations)

fig,axes  = plt.subplots(nrows=len(vdat), figsize=(40,4*len(vdat)),
                             sharex=True)
for ax, data, chan in zip(axes, vdat, vchans):
    ax.scatter((t - t[0])/seconds_per_day, data ,
                   edgecolor='', c='purple', s=3,
                   label=r'$\mathrm{%s}$' % chan.replace('_','\_'),
                   rasterized=True)
    ax.set_yscale('log')
    ax.set_ylim(9, 1.1e4)
    
    ax.set_xlabel('Time (after GPS ' + str(t[0]) + ') [s]')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(EQ_locations)):
        ax.axvline(x=(EQ_locations[e] - t[0])/seconds_per_day,
                       color = 'blue', alpha=0.3, linewidth=2)

plt.xlim((0, (t[-1]-t[0])/seconds_per_day))
fig.tight_layout()

if __debug__:
    print("Saving Figure...")

fig.savefig('Figures/EQ_peaks_indicated.pdf')

# can't have these hard coded path names; doesn't run for anyone else this way
#fig.savefig('/home/roxana.popescu/public_html/'+'EQ_peaks_indicated.png')
