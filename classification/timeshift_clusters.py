# !/usr/bin/env python                                               

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
cl = 4
colors = np.array(['r', 'g', 'b', 'y','c','m','darkgreen','plum','darkblue','pink','orangered','indigo']) 

#read in data
H1dat = loadmat('Data/' + 'H1_SeismicBLRMS.mat')
edat = np.loadtxt('Data/H1_earthquakes.txt')

#read in earthquake channels
'''
cols = [6,12,18,24,30,36,42,48]
vdat = np.array(H1dat['data'][0])
for i in cols:
    add = np.array(H1dat['data'][i])
    vdat = np.vstack((vdat, add))
vchans = np.array(H1dat['chans'][0])
for i in cols:
    vchans = np.append(vchans, H1dat['chans'][i])
vdat_smth = scipy.signal.savgol_filter(vdat,49,1)
'''

#read in six channels
cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
first = 0
num = 18
vdat = np.array(H1dat['data'][first])
for i in cols:
    add = np.array(H1dat['data'][i])
    vdat = np.vstack((vdat, add))
vchans = np.array(H1dat['chans'][first])
for i in cols:
    vchans = np.append(vchans, H1dat['chans'][i])
vdat_smth = scipy.signal.savgol_filter(vdat,49,1)

#shift the data
t_shift = 10 #how many minutes to shift the data by
for i in cols:
    add = np.array(H1dat['data'][i])
    for j in range(1, t_shift+1):        
        add_shift = add[j:]
        #print(np.shape(add_shift))
        add_values = np.zeros((j,1))
        add_shift = np.append(add_shift, add_values)
        #print(np.shape(add_shift))
        vdat = np.vstack((vdat, add_shift))
        chan = 'Time_Shift_' + str(j) + '_Min_EQ_Band_' + str(i)
        vchans = np.append(vchans, chan)
print(np.shape(vdat))
vdat = vdat[:,:43200-t_shift]
print(np.shape(vdat))
#print(vchans)
timetuples = vdat.T
timetuples2 = vdat[0:num].T

#find best number of clusters using calinsky harabaz index

cl=5
kmeans = KMeans(n_clusters=cl-1, random_state=12).fit(timetuples)
kmeans2 = KMeans(n_clusters=cl, random_state=12).fit(timetuples)
kmeans3 = KMeans(n_clusters=cl+1, random_state=12).fit(timetuples)
score1 = metrics.calinski_harabaz_score(timetuples2,kmeans.labels_)
score2 = metrics.calinski_harabaz_score(timetuples2,kmeans2.labels_)
score3 = metrics.calinski_harabaz_score(timetuples2,kmeans3.labels_)
while (not((score1 < score2) and (score3 < score2))):
    cl = cl+1
    print('testing cl = ' + str(cl))
    if (cl == 10):
        break
    kmeans = KMeans(n_clusters=cl-1, random_state=12).fit(timetuples)
    kmeans2 = KMeans(n_clusters=cl, random_state=12).fit(timetuples)
    kmeans3 = KMeans(n_clusters=cl+1, random_state=12).fit(timetuples)
    score1 = metrics.calinski_harabaz_score(timetuples2,kmeans.labels_)
    score2 = metrics.calinski_harabaz_score(timetuples2,kmeans2.labels_)
    score3 = metrics.calinski_harabaz_score(timetuples2,kmeans3.labels_)
print(cl)
print(score2)
print(score1)
print(score3)

kmeans = KMeans(n_clusters=cl, random_state=12).fit(timetuples)
xvals = (np.arange(len(vdat[0])))/(60.*24.)
fig, axes = plt.subplots(len(vdat[0:num]), figsize=(40,4*len(vdat[0:num])))
for ax, data, chan in zip(axes, vdat[0:num], vchans):
    ax.scatter(xvals, data,c=colors[kmeans.labels_],edgecolor='',
               s=3, label=r'$\mathrm{%s}$' % chan.replace('_','\_'))
    ax.set_yscale('log')
    ax.set_ylim(np.median(data)*0.1, max(data)*1.1)
    ax.set_xlim(0,30)
    ax.set_xlabel('Time [days]')
    ax.grid(True, which='both')
    ax.legend()
fig.tight_layout()
fig.savefig('/home/roxana.popescu/public_html/'+'Timeshift_kmeans_XYZchans' + str(cl) + '.png')

'''
#compare with earthquakes
times = '2017-03-01 00:00:00'
t = Time(times,format='iso',scale='utc')
t_start= int(np.floor(t.gps/60)*60)
dur_in_days= 30
dur_in_minutes = dur_in_days*24*60
dur = dur_in_minutes*60
t_end = t_start+dur

#use predicted ground motion to determine which earthquakes are bigger (commented out)    
row, col = np.shape(edat)
gdat = np.array([])
for i in range(row):
    point = edat[i][7]
    gdat = np.append(gdat,point)
gdat = gdat.T
glq = np.percentile(gdat,80)

#use only earthquakes with signifigant ground motion (commented out)                                            
row, col = np.shape(edat)
etime = np.array([])
for i in range(row):
    if (edat[i][7] >= glq):
        point = edat[i][5]
        etime = np.append(etime,point)
#etime = np.array(edat[:,5])                                                                                                              

#use only earthquakes that occur in March 2017                                                                           
col = len(etime)
etime_march = np.array([])
for i in range(col):
    if ((etime[i] >= t_start) and (etime[i] <= t_end)):
        point = etime[i]
        etime_march = np.append(etime_march,point)

#add up number of clusters that appear next to each earthquake                                                                                      
kpoints = np.array([])
#dbpoints = np.array([])
#agpoints = np.array([])                                                                                                                              
#bpoints = np.array([])                                                                                                                                               

xvals = np.arange(t_start,t_end-60*t_shift,60)
for t in etime_march: #for each EQ: collect indices within 5 min of EQ                                                                                                 
    tmin = int(t-5*60)
    tmax = int(t+5*60)
    for j  in range(tmin,tmax):
        val = abs(xvals-j)
        aval = np.argmin(val)
        kpoints = np.append(kpoints, aval)
        #dbpoints  = np.append(dbpoints, aval)
        #agpoints = np.append(agpoints, aval)                                                                                                                         
        #bpoints = np.append(bpoints, aval)                                                                                                                            

kpoints = np.unique(kpoints) #make sure there are no repeating indices                                                                                               
#dbpoints = np.unique(dbpoints)
#agpoints = np.unique(agpoints)                                                                                                                                       
#bpoints = np.unique(bpoints)                                                                                                                                          

kclusters = np.array([])
#dbclusters = np.array([])
#agclusters = np.array([])                                                                                                                                            
#bclusters = np.array([])                                                                                                                                              

for i in kpoints: kclusters = np.append(kclusters,kmeans.labels_[int(i)]) #for each index find the corresponding cluster and store them in array                     
#for i in dbpoints: dbclusters = np.append(dbclusters,db.labels_[int(i)])
#for i in agpoints: agclusters = np.append(agclusters,ag.labels_[int(i)])                                                                                          
#for i in bpoints: bclusters = np.append(bclusters,birch.labels_[int(i)]) 

#kmeans score determined by ratio of points in cluster/points near EQ to  points in cluster/all points                                                           
print('********Results of Kmeans Clustering********')
print('Number of points in each cluster that are near an EQ')
print(collections.Counter(kclusters))
print('Number of points in each cluster')
print(collections.Counter(kmeans.labels_))
k_count = collections.Counter(kclusters).most_common()
ktot_count = collections.Counter(kmeans.labels_).most_common()
k_list_cl = [x[0] for x in k_count]
k_list = [x[1] for x in k_count]
ktot_list_cl = [x[0] for x in ktot_count]
ktot_list = [x[1] for x in ktot_count]
k_clusters = np.array([])
k_compare = np.array([])
k_list2 = np.array([])
ktot_list2 = np.array([])
for i in range(len(k_list_cl)):
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
k_index = np.argmax(k_compare)
#k_max_count = k_list2[0]                                                                                                                 
k_max_count = k_list2[k_index]
k_score = k_max_count/len(kclusters)
#ktot_max_count = ktot_list[0]                                                                                                                      
ktot_max_count = ktot_list2[k_index]
ktot_score = ktot_max_count/len(kmeans.labels_)
krel_score = k_score/ktot_score
print('k_score is '+str(k_score)+', ktot_score is '+str(ktot_score)+ ', and krel_score is ' +str(krel_score))


#plot graph of kmeans clustering for EQ                                                                                          
xvals = np.arange(t_start,t_end-60*t_shift,60)
print(len(xvals))
print(len(vdat[0]))
fig,axes  = plt.subplots(len(vdat[0:num]), figsize=(40,4*len(vdat[0:num])))
for ax, data, chan in zip(axes, vdat[0:num], vchans):
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
fig.savefig('/home/roxana.popescu/public_html/'+'Timeshift_Kmeans_all'+str(cl)+'_.png')                                                                           
'''
