#!/usr/bin/env python

'''
This script reads in seismic noise data from March 2017 and earthquake data.
It shifts the data by time for clustering
It creates a list of earthquake times in March when the peak ground motion is greater than a certain amount. 
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
eps = 2 #min distance for density for dbscan
min_samples=15 #min samples for dbscan

#read in data
H1dat = loadmat('Data/' + 'H1_SeismicBLRMS.mat')
edat = np.loadtxt('Data/H1_earthquakes.txt')

#read in earthquake channels
cols = [6,12,18,24,30,36,42,48]
vdat = np.array(H1dat['data'][0])
vchans = np.array(H1dat['chans'][0])
for i in cols:
    add = np.array(H1dat['data'][i])
    vdat = np.vstack((vdat, add))
for i in cols:
    vchans = np.append(vchans,H1dat['chans'][i])
timetuples = vdat.T

#shift the dat                                                                                                                                             
vdat2 = vdat
vchans2 = vchans
num = 9
t_shift = 30 #how many minutes to shift the data by                                                                                                                 
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

#kmeans clustering loop
for cl in range(2,11):
    kmeans = KMeans(n_clusters=cl, random_state=12).fit(timetuples)
    kpoints = np.array([])
    xvals = np.arange(t_start,t_end,60)
    dbpoints = np.array([])
    for t in etime_march: #for each EQ: collect indices within 5 min of EQ
        tmin = int(t-5*60)
        tmax = int(t+5*60)
        for j  in range(tmin,tmax):
            val = abs(xvals-j)
            aval = np.argmin(val)
            kpoints = np.append(kpoints, aval)
    kpoints = np.unique(kpoints) #make sure there are no repeating indices
    kclusters = np.array([])
    for i in kpoints: kclusters = np.append(kclusters,kmeans.labels_[int(i)]) #for each index find the corresponding cluster and store them in array
    #kmeans score determined by ratio of points in cluster/points near EQ to  points in cluster/all points
    print('  ')
    print('Cl = ' + str(cl))
    print('Number of points in each cluster that are near an EQ')
    print(collections.Counter(kclusters))
    print('Number of points in each cluster')
    print(collections.Counter(kmeans.labels_))
    k_count = collections.Counter(kclusters).most_common()
    ktot_count = collections.Counter(kmeans.labels_).most_common()
    k_list_cl = [x[0] for x in k_count] #cluster number
    k_list = [x[1] for x in k_count] #occurences of cluster
    ktot_list_cl = [x[0] for x in ktot_count]
    ktot_list = [x[1] for x in ktot_count]
    k_clusters = np.array([])
    k_compare = np.array([])
    k_list2 = np.array([])
    ktot_list2 = np.array([])
    for i in range(len(k_list_cl)): #arrange so that k_clusters k_list2 and k_compare are in the same order
        for j in range(len(ktot_list_cl)):
            if k_list_cl[i] == ktot_list_cl[j]:
                k_clusters = np.append(k_clusters,k_list_cl[i])
                compare = k_list[i]/ktot_list[j]
                k_compare = np.append(k_compare, compare)
                k_list2 = np.append(k_list2, k_list[i])
                ktot_list2 = np.append(ktot_list2, k_list[i])
    print('List with the clusters in order')
    print(k_clusters)
    print('Number of points in clusters near EQ divided by total number of points in clusters')
    print(k_compare)
    k_cal_score = metrics.calinski_harabaz_score(timetuples, kmeans.labels_)
    print('For kmeans the calinski harabaz score is ' + str(k_cal_score))
    

#dbscan clustering loop
'''
min_samples_list = [10,20,25,30]
for min_samples in min_samples_list:
    
    db = DBSCAN(eps=eps,min_samples=min_samples).fit(timetuples)

    #print number of clusters
    print(' ')
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print('DBSCAN created ' +str(n_clusters_) + ' clusters')

    #add up number of clusters that appear next to each earthquake
    xvals = np.arange(t_start,t_end,60)
    dbpoints = np.array([])
    for t in etime_march: #for each EQ: collect indices within 5 min of EQ
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
    print('Number of points in each cluster that are near an EQ')
    print(collections.Counter(dbclusters))
    print('Number of points in each cluster')
    print(collections.Counter(db.labels_))
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
    print('List with the clusters in order')
    print(db_clusters)
    print('Number of points in clusters near EQ divided by total number of points in clusters')
    print(db_compare)
    d_cal_score = metrics.calinski_harabaz_score(timetuples, db.labels_)
    print('For dbscan the calinski harabaz score is ' + str(d_cal_score))
'''

#Plot #1: Plot graph of kmeans clustering for EQ
kmeans = KMeans(n_clusters=cl, random_state=12).fit(timetuples)
xvals = np.arange(t_start,t_end,60)
fig,axes  = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, chan in zip(axes, vdat, vchans2):
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
try:
    fig.savefig('/home/roxana.popescu/public_html/'+'EQdata_Kmeans_'+str(cl)+'.png')
except IOerror:
    fig.savefig('Figures/EQdata_Kmeans_'+str(cl)+'.png')

#Plot #2:plot graph of dbscan clustering for EQ
db = DBSCAN(eps=eps,min_samples=min_samples).fit(timetuples)
xvals = np.arange(t_start,t_end,60)
#print number of clusters
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print('DBSCAN created ' +str(n_clusters_) + ' clusters')
fig, axes = plt.subplots(len(vdat), figsize=(40,4*len(vdat)))
for ax, data, chan in zip(axes, vdat, vchans2):
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
try:
    fig.savefig('/home/roxana.popescu/public_html/'+'dbscan_all_.png')
except IOerror:
    fig.savefig('Figures/dbscan_all.png')

