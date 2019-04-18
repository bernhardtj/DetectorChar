#!/usr/bin/env python

'''
This script reads in seismic noise data from March 2017 and earthquake data.
It creates a list of earthquake times in March when the peak ground motion is greater than a certain amount. 
It clusters earthquake channels using kmeans and dbscan.
It compares the clusters around the earthquake times to deterime effectiveness of clustering  
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
cl2 = 3 #number of clusters for agglomerative clustering
cl3 = 7 #number of clusters for birch 
eps = 2 #min distance for density for dbscan
min_samples=15 #min samples for dbscan

#read in data
H1dat = loadmat('Data/' + 'H1_SeismicBLRMS.mat')
edat = np.loadtxt('Data/H1_earthquakes.txt')

# read in earthquake + microseismic channels as vdat/vchannels
'''
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
'''

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

#Plot #1: plot graph of EQ channels with known EQs indicated                                 
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
try:
    fig.savefig('/home/roxana.popescu/public_html/'+'EQs_all_indicated.png')
except FileNotFoundError:
    fig.savefig('Figures/EQs_all_indicated')

#clustering
kmeans = KMeans(n_clusters=cl, random_state=12).fit(timetuples3) #kmeans clustering of earthquake channels
db = DBSCAN(eps=eps,min_samples=min_samples).fit(timetuples3) #dbscan clustering of earthquake channels
#ag = AgglomerativeClustering(n_clusters = cl2, linkage='complete').fit(timetuples3) #agglomerative clustering of earthquake channels
#birch = Birch(n_clusters=cl3).fit(timetuples3)

#other clustering algorithms
#af = AffinityPropagation(preference=preference).fit(timetuples)
#bandwidth = estimate_bandwidth(timetuples, quantile=0.2, n_samples=100)
#ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
#ms.fit(timetuples3)
#labels = spectral_clustering(timetuples, n_clusters=4, eigen_solver='arpack

#print number of clusters
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print('DBSCAN created ' +str(n_clusters_) + ' clusters')
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
bclusters = np.array([])
for i in range(len(a)):
    val = int(a[i])
    kpoint = kmeans.labels_[val]
    kclusters = np.append(kclusters,kpoint)
    dbpoint  = db.labels_[val]
    dbclusters = np.append(dbclusters,dbpoint)
    agpoint = ag.labels_[val]
    agclusters = np.append(agclusters,agpoint)
    bpoint= birch.labels_[val]
    bclusters = np.append(bclusters, bpoint)'''

#add up number of clusters that appear next to each earthquake
kpoints = np.array([])
dbpoints = np.array([])
#agpoints = np.array([])
#bpoints = np.array([])
for t in etime_march: #for each EQ: collect indices within 5 min of EQ
    tmin = int(t-5*60)
    tmax = int(t+5*60)
    for j  in range(tmin,tmax):
        val = abs(xvals-j)
        aval = np.argmin(val)
        kpoints = np.append(kpoints, aval)
        dbpoints  = np.append(dbpoints, aval)
        #agpoints = np.append(agpoints, aval)
        #bpoints = np.append(bpoints, aval)

kpoints = np.unique(kpoints) #make sure there are no repeating indices
dbpoints = np.unique(dbpoints)
#agpoints = np.unique(agpoints)
#bpoints = np.unique(bpoints)

kclusters = np.array([])
dbclusters = np.array([])
#agclusters = np.array([])
#bclusters = np.array([])
for i in kpoints: kclusters = np.append(kclusters,kmeans.labels_[int(i)]) #for each index find the corresponding cluster and store them in array 
for i in dbpoints: dbclusters = np.append(dbclusters,db.labels_[int(i)])
#for i in agpoints: agclusters = np.append(agclusters,ag.labels_[int(i)])
#for i in bpoints: bclusters = np.append(bclusters,birch.labels_[int(i)])
        
#histogram of clusters (commented out)
'''fig = plt.figure()
ax = fig.gca()
ax.hist(kpoints)
fig.savefig('/home/roxana.popescu/public_html/'+'kmeans_'+str(cl)+'_EQ__hist.png'))'''

'''fig = plt.figure()
ax = fig.gca()
ax.hist(dbpoints)
fig.savefig('/home/roxana.popescu/public_html/'+ 'dbscan_EQ__hist.png')'''

#kmeans score determined by ratio of points in cluster/points near EQ to  points in cluster/all points
print('********Results of Kmeans Clustering********')
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
'''
k_index = np.argmax(k_compare)
#k_max_count = k_list2[0]
k_max_count = k_list2[k_index]
k_score = k_max_count/len(kclusters)
#ktot_max_count = ktot_list[0]
ktot_max_count = ktot_list2[k_index]
ktot_score = ktot_max_count/len(kmeans.labels_)
krel_score = k_score/ktot_score
print('k_score is '+str(k_score)+', ktot_score is '+str(ktot_score)+ ', and krel_score is ' +str(krel_score))
'''
#dbscan score determined by percent of points sorted into one cluster near EQ
'''
print('********Results of DBSCAN Clustering********')
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
db_index = np.argmax(db_compare)
#db_max_count = db_list2[0]
db_max_count = db_list2[db_index]
db_score = db_max_count/len(dbclusters)
#dbtot_max_count = dbtot_list[0]
dbtot_max_count = dbtot_list2[db_index]
dbtot_score = dbtot_max_count/len(db.labels_)
dbrel_score = db_score/dbtot_score
print('db_score is '+str(db_score)+', dbtot_score is '+str(dbtot_score)+ ', and dbel_score is ' +str(dbrel_score))
'''

#agglomerative clustering score determined by percent of points sorted into one cluster near EQ
'''print('********Results of Agglomerative Clustering********')
print('Number of points in each cluster that are near an EQ')
print(collections.Counter(agclusters))
print('Number of points in each cluster')
print(collections.Counter(ag.labels_))
ag_count = collections.Counter(agclusters).most_common()
agtot_count = collections.Counter(ag.labels_).most_common()
ag_list_cl = [x[0] for x in ag_count]
ag_list = [x[1] for x in ag_count]
agtot_list_cl = [x[0] for x in agtot_count]
agtot_list = [x[1] for x in agtot_count]
ag_clusters = np.array([])
ag_compare = np.array([])
ag_list2 = np.array([])
agtot_list2 = np.array([])
for i in range(len(ag_list_cl)):
    for j in range(len(ag_list_cl)):
        if ag_list_cl[i] == agtot_list_cl[j]:
            ag_clusters = np.append(ag_clusters,ag_list_cl[i])
            compare = ag_list[i]/agtot_list[j]
            ag_compare = np.append(ag_compare, compare)
            ag_list2 = np.append(ag_list2, ag_list[i])
            agtot_list2 = np.append(agtot_list2, ag_list[i])
print('List with the clusters in order')
print(ag_clusters)
print('Number of points in clusters near EQ divided by total number of points in clusters')
print(ag_compare)
ag_index = np.argmax(ag_compare)
#ag_max_count = ag_list2[0]
ag_max_count = ag_list2[ag_index]
ag_score = ag_max_count/len(agclusters)
#agtot_max_count = agtot_list2[0]
agtot_max_count = agtot_list2[ag_index]
agtot_score = agtot_max_count/len(ag.labels_)
agrel_score = ag_score/agtot_score
print('ag_score is '+str(ag_score)+', agtot_score is '+str(agtot_score)+ ', and agrel_score is ' +str(agrel_score))
'''

#birch clustering score determined by percent of points into one cluster near EQ
'''print('********Results of Birch Clustering********')
print('Number of points in each cluster that are near an EQ')
print(collections.Counter(bclusters))
print('Number of points in each cluster')
print(collections.Counter(birch.labels_))
b_count = collections.Counter(bclusters).most_common()
btot_count = collections.Counter(birch.labels_).most_common()
b_list_cl = [x[0] for x in b_count]
b_list = [x[1] for x in b_count]
btot_list_cl = [x[0] for x in btot_count]
btot_list = [x[1] for x in btot_count]
b_clusters = np.array([])
b_compare = np.array([])
b_list2 = np.array([])
btot_list2 = np.array([])
for i in range(len(b_list_cl)):
    for j in range(len(btot_list_cl)):
        if b_list_cl[i] == btot_list_cl[j]:
            b_clusters = np.append(b_clusters,b_list_cl[i])
            compare = b_list[i]/btot_list[j]
            b_compare = np.append(b_compare, compare)
            b_list2 = np.append(b_list2, b_list[i])
            btot_list2 = np.append(btot_list2, b_list[i])
print('List with the clusters in order')
print(b_clusters)
print('Number of points in clusters near EQ divided by total number of points in clusters')
print(b_compare)
b_index = np.argmax(b_compare)
#b_max_count = b_list2[0]
b_max_count = b_list2[b_index]
b_score = b_max_count/len(bclusters)
#btot_max_count = btot_list2[0]
btot_max_count = btot_list2[b_index]
btot_score = btot_max_count/len(birch.labels_)
brel_score = b_score/btot_score
print('b_score is '+str(b_score)+', btot_score is '+str(btot_score)+ ', and brel_score is ' +str(brel_score))'''

#cluster scores using silhoutette coefficient and calinsky-harabaz index
#print(metrics.silhouette_score(timetuples3,kmeans.labels_))
#print(metrics.silhouette_score(timetuples3,db.labels_))
k_cal_score = metrics.calinski_harabaz_score(timetuples3, kmeans.labels_)
print('For kmeans the calinski harabaz score is ' + str(k_cal_score))
#print(metrics.calinski_harabaz_score(timetuples3, db.labels_))
#print(metrics.calinsku_harabaz_score(timetuples3, ag.labels_))
#print(metrics.calinski_harabaz_score(timetuples3, birch.labels_))

#Plot #2: Plot graph of kmeans clustering for EQ
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
try:
    fig.savefig('/home/roxana.popescu/public_html/'+'EQdata_Kmeans_'+str(cl)+'_.png')
except FileNotFoundError:
    fig.savefig('Figures/EQdata_Kmeans_'+str(cl)+'.png')

#Plot #3:plot graph of dbscan clustering for EQ
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
try:
    fig.savefig('/home/roxana.popescu/public_html/'+'dbscan.png')
except FileNotFoundError:
    fig.savefig('Figures/dbscan.png')

#Plot #4: plot agglomerative clustering graph for EQ
'''
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
try:
    fig.savefig('/home/roxana.popescu/public_html/'+'EQ_agclustering_'+str(cl2)+'.png')
except FileNotFoundError:
    fig.savefig('Figures/EQ_agclustering_'+str(cl2)+'.png')'''

#Plot #5: plot birch  clustering graph for EQ
'''
fig,axes  = plt.subplots(len(vdat2), figsize=(40,4*len(vdat2)))
for ax, data, chan in zip(axes, vdat2, vchans2):
    ax.scatter(xvals, data,c=colors[birch.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
    for e in range(len(etime_march)):
        ax.axvline(x=etime_march[e])
fig.tight_layout()
try:
    fig.savefig('/home/roxana.popescu/public_html/'+'EQ_birchclustering_'+str(cl3)+'.png')
except FileNotFoundError:
    fig.savefig('Figures/EQ+microseism_birchclustering_'+str(cl3)+'.png')
'''
