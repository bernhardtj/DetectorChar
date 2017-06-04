#!/usr/bin/env python

from __future__ import division
from sklearn.cluster import KMeans
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('axes', labelsize=20.0)
plt.rc('axes', axisbelow=True)
plt.rc('axes.formatter', limits=[-3,4])
plt.rc('legend', fontsize=14.0)
plt.rc('xtick', labelsize=16.0)
plt.rc('ytick', labelsize=16.0)
plt.rc('figure', dpi=100)

H1dat      = loadmat('Data/' + 'H1_SeismicBLRMS_March.mat')
vdat       = H1dat['data'][-6:]
vchans     = H1dat['chans'][-6:]
timetuples = vdat.T

cl = 6
colors = np.array(['r', 'g', 'b', 'y', 'c', 'm'])

kmeans = KMeans(n_clusters=cl, random_state=12).fit(timetuples)

xvals = (np.arange(len(vdat[0])))/(60.*24.)

fig, axes = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, chan in zip(axes, vdat, vchans):
    ax.scatter(xvals, data, c=colors[kmeans.labels_], edgecolor=None, 
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlim(0,30)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
fig.tight_layout()
try:
    fig.savefig('Figures/' + 'sei-test3.png')
except RuntimeError as e:
    if 'latex' in str(e).lower():
        fig.savefig('Figures/' + 'sei-test3.png')
    else:
        raise
