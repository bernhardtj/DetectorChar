#!/usr/bin/env python

from __future__ import division
from sklearn.cluster import KMeans
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.signal

plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('axes', labelsize=20.0)
plt.rc('axes', axisbelow=True)
plt.rc('axes.formatter', limits=[-3,4])
plt.rc('legend', fontsize=14.0)
plt.rc('xtick', labelsize=16.0)
plt.rc('ytick', labelsize=16.0)
plt.rc('figure', dpi=100)

#variables
w = 49 #window length for savgol filter
p = 1 #polynomial order for savgol filter
cl = 4 #number of clusters
colors = np.array(['r', 'g', 'b', 'y']) #colors to assign to each cluster

#read in data
H1dat = loadmat('Data/' + 'H1_SeismicBLRMS_March.mat')

#read in all earthquake data/channels into vdat/vchans
cols = [6,12]
vdat = np.array(H1dat['data'][0])
for i in cols:
    add = np.array(H1dat['data'][i])
    vdat = np.vstack((vdat, add))
vchans = np.array(H1dat['chans'][0])
for i in cols:
    vchans = np.append(vchans,H1dat['chans'][i])

#apply smoothing filter to data
vdat_smth = scipy.signal.savgol_filter(vdat,w,p)
#take derivative of smoothed data
vdat_diff = np.diff(vdat_smth)
#transpose vdat_diff
timetuples = vdat_diff.T

#add original data and deriv data together into vdat/vchans_both
vdat_diff2 = np.hstack((vdat_diff,[[0],[0],[0]]))
vboth = np.vstack((vdat,vdat_diff2))
vchans_both = np.append(vchans,vchans)
timetuples2 = vboth.T

#take the square of the derivatives as vdat_diff3/vchans
vdat_diff3 = np.square(vdat_diff)
timetuples3 = vdat_diff3.T

#kmeans clustering of the data 
kmeans = KMeans(n_clusters=cl, random_state=12).fit(timetuples) #cluster according to derivatives
kmeans2 = KMeans(n_clusters=cl, random_state=12).fit(timetuples2) #cluster according to both original data and derivatives
kmeans3 = KMeans(n_clusters=cl, random_state=12).fit(timetuples3) #cluster according to square of derivatives

#plot of original values (and smoothed data) that are clustered according to derivatives
xvals = (np.arange(len(vdat[0])))/(60.*24.)
fig,axes  = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, data2, chan in zip(axes, vdat, vdat_smth, vchans):
    ax.scatter(xvals, data,c=colors[kmeans.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.plot(xvals,data2)
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlim(0,30)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
fig.tight_layout()
#fig.savefig('/home/roxana.popescu/public_html/'+'EQ_XYZ_'+ str(cl)+'_data_deriv.png')
fig.savefig('Figures/EQ_XYZ_' + str(cl) + '_data_deriv.png')

#plot of derivatives that are clustered according to derivatives 
xvals = (np.arange(len(vdat_diff[0])))/(60.*24.)
fig,axes  = plt.subplots(len(vdat_diff), figsize=(40,4*len(vdat_diff)))
for ax, data, chan in zip(axes, vdat_diff, vchans):
    ax.scatter(xvals, data,c=colors[kmeans.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    #ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlim(0,30)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
fig.tight_layout()
#fig.savefig('/home/roxana.popescu/public_html/'+'EQ_XYZ_'+ str(cl)+'_deriv_deriv.png')
fig.savefig('Figures/EQ_XYZ_' + str(cl) + '_deriv_data.png')

#plot of original values (and smoothed data) that are clustered according to original data and derivatives
xvals = (np.arange(len(vdat[0])))/(60.*24.)
fig,axes  = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, data2, chan in zip(axes, vdat, vdat_smth, vchans):
    ax.scatter(xvals, data,c=colors[kmeans2.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.plot(xvals,data2)
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlim(0,30)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
fig.tight_layout()
#fig.savefig(os.path.join('/home/roxana.popescu/public_html/','EQ_XYZ_'+ str(cl)+'_data_data+deriv.png')
fig.savefig('Figures/EQ_XYZ_' + str(cl) + '_data_data+deriv.png')

#plot of square of derivatives that are clustered according to square of derivatives
xvals = (np.arange(len(vdat_diff3[0])))/(60.*24.)
fig,axes  = plt.subplots(len(vdat_diff3), figsize=(40,4*len(vdat_diff3)))
for ax, data, chan in zip(axes, vdat_diff3, vchans):
    ax.scatter(xvals, data,c=colors[kmeans3.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlim(0,30)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
fig.tight_layout()
#fig.savefig('/home/roxana.popescu/public_html/'+'EQ_XYZ_'+ str(cl)+'_deriv2_deriv2.png')
fig.savefig('Figures/EQ_XYZ_' + str(cl) + 'deriv2_deriv2.png')

#plot of original values (and smoothed data) that are clustered according to square of derivatives
xvals = (np.arange(len(vdat[0])))/(60.*24.)
fig,axes  = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, data2, chan in zip(axes, vdat, vdat_smth, vchans):
    ax.scatter(xvals, data,c=colors[kmeans3.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.plot(xvals,data2)
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlim(0,30)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
fig.tight_layout()
#fig.savefig('/home/roxana.popescu/public_html/'+'EQ_XYZ_'+ str(cl)+'_data_deriv2.png')
fig.savefig('Figures/EQ_XYZ_'+str(cl)+'_data_deriv2.png')
