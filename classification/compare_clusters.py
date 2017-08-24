#!/usr/bin/env python

'''
This script reads in seismic noise data from March 2017 and earthquake data.
It shifts the data by time for clustering
It determined earthquake times by looking at peaks in data
It clusters earthquake channels using kmeans and dbscan.
It compares the clusters around the earthquake times to deterime effectiveness of clustering
It plots the data as clustered by kmeans and dbscan
'''

from __future__ import division
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift,estimate_bandwidth
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.signal as sig
from astropy.time import Time
import collections

plt.rc('text',   usetex = True)
plt.rc('font',   **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('axes',   labelsize = 20.0)
plt.rc('axes',   axisbelow = True)
plt.rc('axes.formatter', limits=[-3,4])
plt.rc('legend', fontsize  = 14.0)
plt.rc('xtick',  labelsize = 16.0)
plt.rc('ytick',  labelsize = 16.0)
plt.rc('figure', dpi = 100)

# colors for clusters
colors = np.array(['r', 'g', 'b','y','c','m','darkgreen','plum',
                       'darkblue','pink','orangered','indigo'])
cl          = 6   # number of clusters for kmeans
eps         = 2   # min distance for density for DBscan
min_samples = 15  # min samples for DBscan

#read in data
H1dat = loadmat('Data/' + 'H1_SeismicBLRMS.mat')
#edat  = np.loadtxt('Data/H1_earthquakes.txt')

# read in earthquake channels
cols   = [6,12,18,24,30,36,42,48]      # NEED comment here
vdat   = np.array(H1dat['data'][0])
vchans = np.array(H1dat['chans'][0])
for i in cols:
    add = np.array(H1dat['data'][i])
    vdat = np.vstack((vdat, add))
for i in cols:
    vchans = np.append(vchans,H1dat['chans'][i])
timetuples = vdat.T

# shift the data
vdat2   = vdat
vchans2 = vchans
num     = 10
t_shift = 10 # how many minutes to shift the data by
for i in cols:
    add = np.array(H1dat['data'][i])
    for j in range(1, t_shift+1):
        add_shift = add[j:]
        add_values = np.zeros((j,1))
        add_shift = np.append(add_shift, add_values)
        vdat2 = np.vstack((vdat2, add_shift))
        chan = 'Time_Shift_' + str(j) + '_Min_EQ_Band_' + str(i)
        vchans2 = np.append(vchans2, chan)
print(np.shape(vdat2))
vdat2 = vdat[:,:43200-t_shift]
print(np.shape(vdat2))
timetuples2 = vdat.T
timetuples3 = vdat[0:num].T

 #convert time to gps time
times       = '2017-03-01 00:00:00'
ti           = Time(times,format='iso',scale='utc')
t_start     = int(np.floor(ti.gps/60)*60)
dur_in_days = 30
dur_in_minutes = dur_in_days*24*60
dur         = dur_in_minutes*60
t_end       = t_start + dur
t    = np.arange(t_start, t_end, 60)

# create list of earthquake times from peaks
# find peaks in all three z channel directions
widths  = np.arange(5, 140)   # range of widths in minutes
min_snr = 5
noise_perc = 15
peaks1 = sig.find_peaks_cwt(vdat[2], widths,
                                min_snr = min_snr, noise_perc=noise_perc)
peaks2 = sig.find_peaks_cwt(vdat[5], widths,
                                min_snr = min_snr, noise_perc=noise_perc)
peaks3 = sig.find_peaks_cwt(vdat[8], widths,
                                min_snr = min_snr, noise_perc=noise_perc)

# takes average time for earthquake times from three channels
# that are within dtau minutes of each other 
dtau = 3
peak_list = np.array([])
for i in peaks1:
    for j in peaks2:
        for k in peaks3:
            if (abs(i-j) <= dtau and abs(i-k) <= dtau):
                avg = (i+j+k)/3
                peak_list = np.append(peak_list, avg)
EQ_times = np.array([])
for i in peak_list:
    EQ_times = np.append(EQ_times, t[int(i)])

# kmeans clustering loop
'''
Nmin = 2
num = 9
Nmax = Nmin + num  
for cl in range(Nmin, Nmax):
    kmeans   = KMeans(n_clusters=cl, random_state=13).fit(timetuples2)
    kpoints  = np.array([])
    xvals    = np.arange(t_start, t_end, 60)
    for t in EQ_times: #for each EQ: collect indices within 30 min of EQ
        tmin = int(t - 10)
        tmax = int(t + 10)
        for j  in range(tmin, tmax):
            val     = abs(xvals - j)
            aval    = np.argmin(val)
            kpoints = np.append(kpoints, aval)
    kpoints   = np.unique(kpoints) # make sure there are no repeating indices
    kclusters = np.array([])
    for i in kpoints:
        #for each index find the corresponding cluster and store them in array
        kclusters = np.append(kclusters, kmeans.labels_[int(i)])
        # kmeans score determined by ratio of points in
        # cluster/points near EQ to  points in cluster/all points
    k_count      = collections.Counter(kclusters).most_common()
    ktot_count   = collections.Counter(kmeans.labels_).most_common()
    k_list_cl    = [x[0] for x in k_count] #cluster number
    k_list       = [x[1] for x in k_count] #occurences of cluster
    ktot_list_cl = [x[0] for x in ktot_count]
    ktot_list    = [x[1] for x in ktot_count]
    k_clusters   = np.array([])
    k_compare    = np.array([])
    k_list2      = np.array([])
    ktot_list2   = np.array([])
    # arrange so that k_clusters k_list2 and k_compare are in the same order
    for i in range(len(k_list_cl)):
        for j in range(len(ktot_list_cl)):
            if k_list_cl[i] == ktot_list_cl[j]:
                k_clusters = np.append(k_clusters,k_list_cl[i])
                compare    = k_list[i]/ktot_list[j]
                k_compare  = np.append(k_compare, compare)
                k_list2    = np.append(k_list2, k_list[i])
                ktot_list2 = np.append(ktot_list2, k_list[i])
    np.set_printoptions(precision=3)
    #print(k_clusters)
    #print(k_compare)
    max_val = max(k_compare)
    max_index = np.argmax(k_compare)
    max_cluster = int(k_clusters[max_index])
    k_cal_score = metrics.calinski_harabaz_score(timetuples, kmeans.labels_)
    #print('K-means ' + str(cl) + ':  C-H score = {:0.6g}'.format(k_cal_score))
    print(str(cl) + ' & {:0.6g}'.format(k_cal_score) + ' & ' + str(max_cluster) + ' & {:0.6g}'.format(max_val))
    print('\\hline')
'''

# dbscan clustering loop
'''
min_samples_list = [10,20,25,30]
eps_list  = [1,2,3,4,5]
#for min_samples in min_samples_list:
for eps in eps_list:
    db = DBSCAN(eps=eps,min_samples=min_samples).fit(timetuples)

    #number of clusters
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    #add up number of clusters that appear next to each earthquake
    xvals = np.arange(t_start,t_end,60)
    dbpoints = np.array([])
    for t in EQ_times: #for each EQ: collect indices within 5 min of EQ
        tmin = int(t-5*60)
        tmax = int(t+5*60)
        for j  in range(tmin,tmax):
            val = abs(xvals-j)
            aval = np.argmin(val)
            dbpoints  = np.append(dbpoints, aval)

    dbpoints = np.unique(dbpoints)
    dbclusters = np.array([])

    for i in dbpoints: dbclusters = np.append(dbclusters,db.labels_[int(i)]) #for each index find the corresponding cluster and store them in array

    #dbscan score determined by percent of points sorted into one cluster near EQ
    db_count = collections.Counter(dbclusters).most_common()
    dbtot_count = collections.Counter(db.labels_).most_common()
    db_list_cl = [x[0] for x in db_count]
    db_list = [x[1] for x in db_count]
    dbtot_list_cl = [x[0] for x in dbtot_count]
    dbtot_list = [x[1] for x in dbtot_count]
    db_clusters = np.array([])
    db_compare = np.array([])
    db_list2 = np.array([])
    dbtot_list2 = np.array([])
    for i in range(len(db_list_cl)):
        for j in range(len(dbtot_list_cl)):
            if db_list_cl[i] == dbtot_list_cl[j]:
                db_clusters = np.append(db_clusters,db_list_cl[i])
                compare = db_list[i]/dbtot_list[j]
                db_compare = np.append(db_compare, compare)
                db_list2 = np.append(db_list2, db_list[i])
                dbtot_list2 = np.append(dbtot_list2, db_list[i])
    #print(db_clusters)
    #print(db_compare)
    max_val = max(db_compare)
    max_index = np.argmax(db_compare)
    max_cluster = int(db_clusters[max_index])
    db_cal_score = metrics.calinski_harabaz_score(timetuples, db.labels_)
    print(str(eps) + ' & ' +  str(min_samples) + ' & ' +  str(n_clusters) + ' & {:0.6g}'.format(db_cal_score) + ' & ' + str(max_cluster) + ' & {:0.6g}'.format(max_val))
    print('\\hline')
'''   
 
#ag clustering loop
'''
Nmin = 2
num = 9
Nmax = Nmin + num  
for cl in range(Nmin, Nmax):
    ag   = AgglomerativeClustering(n_clusters=cl).fit(timetuples)
    agpoints  = np.array([])
    xvals    = np.arange(t_start, t_end, 60)
    for t in EQ_times: #for each EQ: collect indices within 30 min of EQ
        tmin = int(t - 10)
        tmax = int(t + 10)
        for j  in range(tmin, tmax):
            val     = abs(xvals - j)
            aval    = np.argmin(val)
            agpoints = np.append(agpoints, aval)
    agpoints   = np.unique(agpoints) # make sure there are no repeating indices
    agclusters = np.array([])
    for i in agpoints:
        #for each index find the corresponding cluster and store them in array
        agclusters = np.append(agclusters, ag.labels_[int(i)])
    ag_count      = collections.Counter(agclusters).most_common()
    agtot_count   = collections.Counter(ag.labels_).most_common()
    ag_list_cl    = [x[0] for x in ag_count] #cluster number
    ag_list       = [x[1] for x in ag_count] #occurences of cluster
    agtot_list_cl = [x[0] for x in agtot_count]
    agtot_list    = [x[1] for x in agtot_count]
    ag_clusters   = np.array([])
    ag_compare    = np.array([])
    ag_list2      = np.array([])
    agtot_list2   = np.array([])
    # arrange so that k_clusters k_list2 and k_compare are in the same order
    for i in range(len(ag_list_cl)):
        for j in range(len(agtot_list_cl)):
            if ag_list_cl[i] == agtot_list_cl[j]:
                ag_clusters = np.append(ag_clusters,ag_list_cl[i])
                compare    = ag_list[i]/agtot_list[j]
                ag_compare  = np.append(ag_compare, compare)
                ag_list2    = np.append(ag_list2, ag_list[i])
                agtot_list2 = np.append(agtot_list2, ag_list[i])
    np.set_printoptions(precision=3)
    max_val = max(ag_compare)
    max_index = np.argmax(ag_compare)
    max_cluster = int(ag_clusters[max_index])
    ag_cal_score = metrics.calinski_harabaz_score(timetuples, ag.labels_)
    print(str(cl) + ' & {:0.6g}'.format(ag_cal_score) + ' & ' + str(max_cluster) + ' & {:0.6g}'.format(max_val))
    print('\\hline')
'''

#birch clustering loop
'''
Nmin = 2
num = 9
Nmax = Nmin + num  
for cl in range(Nmin, Nmax):
    birch   = Birch(n_clusters=cl).fit(timetuples)
    bpoints  = np.array([])
    xvals    = np.arange(t_start, t_end, 60)
    for t in EQ_times: #for each EQ: collect indices within 30 min of EQ
        tmin = int(t - 10)
        tmax = int(t + 10)
        for j  in range(tmin, tmax):
            val     = abs(xvals - j)
            aval    = np.argmin(val)
            bpoints = np.append(bpoints, aval)
    bpoints   = np.unique(bpoints) # make sure there are no repeating indices
    bclusters = np.array([])
    for i in bpoints:
        #for each index find the corresponding cluster and store them in array
        bclusters = np.append(bclusters, birch.labels_[int(i)])
    b_count      = collections.Counter(bclusters).most_common()
    btot_count   = collections.Counter(birch.labels_).most_common()
    b_list_cl    = [x[0] for x in b_count] #cluster number
    b_list       = [x[1] for x in b_count] #occurences of cluster
    btot_list_cl = [x[0] for x in btot_count]
    btot_list    = [x[1] for x in btot_count]
    b_clusters   = np.array([])
    b_compare    = np.array([])
    b_list2      = np.array([])
    btot_list2   = np.array([])
    # arrange so that k_clusters k_list2 and k_compare are in the same order
    for i in range(len(b_list_cl)):
        for j in range(len(btot_list_cl)):
            if b_list_cl[i] == btot_list_cl[j]:
                b_clusters = np.append(b_clusters, b_list_cl[i])
                compare    = b_list[i]/btot_list[j]
                b_compare  = np.append(b_compare, compare)
                b_list2    = np.append(b_list2, b_list[i])
                btot_list2 = np.append(btot_list2, b_list[i])
    np.set_printoptions(precision=3)
    max_val = max(b_compare)
    max_index = np.argmax(b_compare)
    max_cluster = int(b_clusters[max_index])
    b_cal_score = metrics.calinski_harabaz_score(timetuples, birch.labels_)
    print(str(cl) + ' & {:0.6g}'.format(b_cal_score) + ' & ' + str(max_cluster) + ' & {:0.6g}'.format(max_val))
    print('\\hline')
'''

#hdbscan?

# Plot 1:plot graph of kmeans clustering for EQ
kmeans = KMeans(n_clusters=cl, random_state=12).fit(timetuples)
xvals = np.arange(t_start, t_end, 60)
fig,axes  = plt.subplots(len(vdat[0:3]), figsize=(40, 4*len(vdat[0:3])))
for ax, data, chan in zip(axes, vdat[0:3], vchans2):
    ax.scatter(xvals, data,
                   c = colors[kmeans.labels_],
                   edgecolor = '',
               s=4, alpha=0.8, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(8, 11000)
    ax.set_xlabel('GPS Time')
    ax.set_ylabel('RMS velocity [nm/s]')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(EQ_times)):
        ax.axvline(x=EQ_times[e])
fig.tight_layout()
fig.savefig('Figures/EQdata_Kmeans_' + str(cl) + '.png',
                rasterized=True)
try:
    fig.savefig('/home/roxana.popescu/public_html/' + 'EQdata_Kmeans_'+str(cl)+'.png',
                    rasterized=True)
except:
    print(" ")

kmeans = KMeans(n_clusters=cl, random_state=12).fit(timetuples)
xvals = np.arange(t_start, t_end, 60)
num1 = 1.172 *pow(10,9)
num2 = 2.5*pow(10,6)
num3 = 3*pow(10,6)
fig,axes  = plt.subplots(len(vdat[0:3]), figsize=(10, 4*len(vdat[0:3])))
for ax, data, chan in zip(axes, vdat[0:3], vchans2):
    ax.scatter(xvals, data,
                   c = colors[kmeans.labels_],
                   edgecolor = '',
               s=4, alpha=0.8, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(8, 11000)
    ax.set_xlim(num1+num2, num1+num3)
    ax.set_xlabel('GPS Time')
    ax.set_ylabel('RMS velocity [nm/s]')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(EQ_times)):
        ax.axvline(x=EQ_times[e])
fig.tight_layout()
try:
    fig.savefig('/home/roxana.popescu/public_html/' + 'EQdata_Kmeans_'+str(cl)+'_crop.png',
                    rasterized=True)
except:
    print(" ")

# Plot 2:plot graph of dbscan clustering for EQ
'''
db    = DBSCAN(eps=eps,min_samples=min_samples).fit(timetuples)
xvals = np.arange(t_start, t_end, 60)
# print number of clusters
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print('DBSCAN created ' +str(n_clusters_) + ' clusters')
fig, axes = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, chan in zip(axes, vdat, vchans2):
    ax.scatter(xvals, data, c=colors[db.labels_], edgecolor='',
               s=5, alpha=0.8, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(8, 11000)
    ax.set_xlabel('GPS Time')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(etime_march)):
        ax.axvline(x=etime_march[e])
fig.tight_layout()
fig.savefig('Figures/dbscan_all.png',
                rasterized=True)
try:
    fig.savefig('/home/roxana.popescu/public_html/' + 'dbscan_all_.png',
                    rasterized=True)
except:
    print(" ")
'''
