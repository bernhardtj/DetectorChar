#!/usr/bin/env python

from __future__ import division
from sklearn.cluster import KMeans
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

plt.rc('text', usetex=False)
#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('axes', labelsize=22.0)
plt.rc('axes', axisbelow=True)
plt.rc('axes.formatter', limits=[-3,4])
plt.rc('legend', fontsize=18.0)
plt.rc('xtick', labelsize=18.0)
plt.rc('ytick', labelsize=18.0)
plt.rc('figure', dpi=200)

H1dat      = loadmat('Data/' + 'H1_SeismicBLRMS_March.mat')
vdat       = H1dat['data'][-6:]
vchans     = H1dat['chans'][-6:]
timetuples = vdat.T

N_clusters = 6
colors = np.array(['r', 'g', 'b', 'y', 'c', 'm'])

print("K-Means Clustering...")
kmeans = KMeans(n_clusters=N_clusters, random_state=12).fit(timetuples)

xvals = (np.arange(len(vdat[0])))/(60*24)

print("Plotting...")
fig, axes = plt.subplots(len(vdat), figsize=(40, 4*len(vdat)))
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
print("Saving figs...")
figname = 'SeismicBLRMS_Kmeans'
try:
    fig.savefig('Figures/' + figname)
except RuntimeError as e:
    if 'latex' in str(e).lower():
        fig.savefig('Figures/' + figname)
    else:
        raise
