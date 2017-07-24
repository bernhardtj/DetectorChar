#!/usr/bin/env python

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
import scipy.signal
from astropy.time import Time
import collections

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
colors = np.array(['r', 'g', 'b', 'y','c','m','darkgreen','plum','darkblue','pink','orangered','indigo']) #colors for clusters
cl= 6 #number of clusters for kmeans
cl2 = 4 #number of clusters for agglomerative clustering
cl3 = 5 #number of clusters for 
eps = 2 #min distance for density for dbscan
min_samples=15 #min samples for dbscan

#read in data
H1dat = loadmat('Data/' + 'H1_SeismicBLRMS.mat')
edat = np.loadtxt('Data/H1_earthquakes.txt')

# read in earthquake + microseismic channels as vdat/vchannels
cols = [1,6,7,12,13,18,19,24,25,30,31,36,37,42,43,48,49]
vdat = np.array(H1dat['data'][0])
for i in cols:
    add = np.array(H1dat['data'][i])
    vdat = np.vstack((vdat, add))
vchans = np.array(H1dat['chans'][0])
for i in cols:
    vchans = np.append(vchans,H1dat['chans'][i])
timetuples = vdat.T
vdat_smth = scipy.signal.savgol_filter(vdat,49,1)
timetuples2 = vdat_smth.T

#read in earthquake channels as vdat2/vchan2
cols = [6,12,18,24,30,36,42,48]
vdat2 = np.array(H1dat['data'][0])
for i in cols:
    add = np.array(H1dat['data'][i])
    vdat2 = np.vstack((vdat2, add))
vchans2 = np.array(H1dat['chans'][0])
for i in cols:
    vchans2 = np.append(vchans2,H1dat['chans'][i])
timetuples3 = vdat2.T
vdat_smth2 = scipy.signal.savgol_filter(vdat2,49,1)
timetuples4 = vdat_smth2.T

#use predicted ground motion to determine which earthquakes are bigger (commented out)
'''col = len(edat)
gdat = np.array([])
for i in range(col):
    point = edat[i][7]
    gdat = np.append(gdat,point)
gdat = gdat.T
glq = np.percentile(gdat,83.75)'''

#convert time to gps time                                                                      
times = '2017-03-01 00:00:00'
t = Time(times,format='iso',scale='utc')
t_start= int(np.floor(t.gps/60)*60)
dur_in_days= 30
dur_in_minutes = dur_in_days*24*60
dur = dur_in_minutes*60
t_end = t_start+dur

#use only earthquakes with signifigant ground motion (commented out)                         
'''col = len(edat)
etime = np.array([])
for i in range(col):
    if (edat[i][7] >= glq):
        point = edat[i][5]
        etime = np.append(etime,point)'''
etime = np.array(edat[:,5])

#use only earthqaukes that occur in March 2017         
col = len(etime)
etime_march = np.array([])
for i in range(col):
    if ((etime[i] >= t_start) and (etime[i] <= t_end)):
        point = etime[i]
        etime_march = np.append(etime_march,point)

#plot graph of EQ channels with known EQs indicated                                 
xvals = np.arange(t_start,t_end,60)
fig,axes  = plt.subplots(len(vdat2),figsize=(40,4*len(vdat2)))
for ax, data, chan in zip(axes, vdat2, vchans2):
    ax.scatter(xvals, data, c = 'k',edgecolor='',
                      s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlabel('GPS Time')
    ax.grid(True, which='both')
    ax.legend()
    for e in etime_march:
        ax.axvline(x=e)
fig.tight_layout()
#fig.savefig('/home/roxana.popescu/public_html/'+'EQs_all_indicated.png')
fig.savefig('Figures/EQs_all_indicated')

#clustering
kmeans = KMeans(n_clusters=cl, random_state=12).fit(timetuples3) #kmeans clustering of earthquake channels
db = DBSCAN(eps=eps,min_samples=min_samples).fit(timetuples3) #dbscan clustering of earthquake channels
ag = AgglomerativeClustering(n_clusters = cl2, linkage='complete').fit(timetuples3) #agglomerative clustering of earthquake channels

#other clustering algorithms
#birch = Birch(n_clusters=cl3).fit(timetuples)
#af = AffinityPropagation(preference=preference).fit(timetuples)
#bandwidth = estimate_bandwidth(timetuples, quantile=0.2, n_samples=100)
#ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
#ms.fit(timetuples3)
#labels = spectral_clustering(timetuples, n_clusters=4, eigen_solver='arpack

#print number of clusters
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print('there are ' +str(n_clusters_) + ' clusters for dbscan')
#n_clusters2 = len(set(af.labels_)) - (1 if -1 in af.labels_ else 0)
#print('there are ' +str(n_clusters2) + ' clusters for affinity propagatiom')
#n_clusters3 = len(set(ms.labels_)) - (1 if -1 in ms.labels_ else 0)
#print('there are ' +str(n_clusters3) + ' clusters for meanshift')

#find cluster labels at each earthquake occurence (commented out)
'''a = np.array([])
for i in etime_march:
    vals = abs(xvals - i)
    aval = np.argmin(vals)
    a = np.append(a,aval)
       
kclusters = np.array([])
dbclusters = np.array([])
agclusters = np.array([])
for i in range(len(a)):
    val = int(a[i])
    kcluster = kmeans.labels_[val]
    kclusters = np.append(kclusters,kcluster)
    dbcluster  = db.labels_[val]
    dbclusters = np.append(dbclusters,dbcluster)
    agcluster = ag.labels_[val]
    agclusters = np.append(agclusters,agcluster)
'''
#add up number of clusters that appear next to each earthquake
kclusters = np.array([])
dbclusters = np.array([])
agclusters = np.array([])
for t in etime_march:
    tmin = int(t-5*60)
    tmax = int(t+5*60)
    for j  in range(tmin,tmax):
        val = abs(xvals-j)
        aval = np.argmin(val)
        kcluster = kmeans.labels_[aval]
        kclusters = np.append(kclusters,kcluster)
        dbcluster  = db.labels_[aval]
        dbclusters = np.append(dbclusters,dbcluster)
        agcluster = ag.labels_[aval]
        agclusters = np.append(agclusters,agcluster)

#histogram of clusters (commented out)                                                                                                                            

'''fig = plt.figure()
ax = fig.gca()
ax.hist(kclusters)
fig.savefig('/home/roxana.popescu/public_html/'+'kmeans_'+str(cl)+'_EQ__hist.png'))'''

'''fig = plt.figure()
ax = fig.gca()
ax.hist(dbclusters)
fig.savefig('/home/roxana.popescu/public_html/'+dbscan_EQ__hist.png')'''

#kmeans score determined by percent of points sorted into one cluster near EQ
print(collections.Counter(kclusters))
k_count = collections.Counter(kclusters).most_common(1)
k_list = [x[1] for x in k_count]
k_max = k_list[0]
k_score = k_max/len(kclusters)
print('kscore is ' + str(k_score))

#dbscan score determined by percent of points sorted into one cluster near EQ
print(collections.Counter(dbclusters))
db_count = collections.Counter(dbclusters).most_common(1)
db_list = [x[1] for x in db_count]
db_max = db_list[0]
db_score = db_max/len(dbclusters)
print('dbscore is ' + str(db_score))

#agglomerative clustering score determined by percent of points sorted into one cluster near EQ
print(collections.Counter(agclusters))
ag_count = collections.Counter(agclusters).most_common(1)
ag_list = [x[1] for x in ag_count]
ag_max = ag_list[0]
ag_score = ag_max/len(agclusters)
print('agscore is ' + str(ag_score))

#plot graph of kmeans clustering for EQ
xvals = np.arange(t_start,t_end,60)
fig,axes  = plt.subplots(len(vdat2), figsize=(40,4*len(vdat2)))
for ax, data, chan in zip(axes, vdat2, vchans2):
    ax.scatter(xvals, data,c=colors[kmeans.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlabel('GPS Time')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(etime_march)):
        ax.axvline(x=etime_march[e])
fig.tight_layout()
#fig.savefig('/home/roxana.popescu/public_html/'+'Kmeans_all'+str(cl)+'_.png')
fig.savefig('Figures/Kmeans_all_'+str(cl)+'.png')

#plot graph of dbscan clustering for EQ
fig, axes = plt.subplots(len(vdat2), figsize=(40,4*len(vdat2)))
for ax, data, chan in zip(axes, vdat2, vchans2):
    ax.scatter(xvals, data, c=colors[db.labels_], edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlabel('GPS Time')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(etime_march)):
        ax.axvline(x=etime_march[e])
fig.tight_layout()
#fig.savefig('/home/roxana.popescu/public_html/'+'dbscan_all_.png')
fig.savefig('Figures/dbscan_all.png')

#plot agglomerative clustering graph for EQ
fig,axes  = plt.subplots(len(vdat2), figsize=(40,4*len(vdat2)))
for ax, data, chan in zip(axes, vdat2, vchans2):
    ax.scatter(xvals, data,c=colors[ag.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(etime_march)):
        ax.axvline(x=etime_march[e])
fig.tight_layout()
#fig.savefig('/home/roxana.popescu/public_html/'+'EQ_agclustering_'+str(cl2)+'.png')
fig.savefig('Figures/EQ_agclustering_'+str(cl2)+'.png')

'''
#plot birch  clustering graph for EQ+microseism
fig,axes  = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, chan in zip(axes, vdat, vchans):
    ax.scatter(xvals, data,c=colors[birch.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlim(0,30)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
fig.tight_layout()
#fig.savefig('/home/roxana.popescu/public_html','EQ+microseism_birchclustering_'+str(cl3)+'.png')
fig.savefig('Figures/EQ+microseism_birchclustering_'+str(cl3)+'.png')
'''
