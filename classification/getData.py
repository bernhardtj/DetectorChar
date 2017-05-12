# this function gets some data (from the 40m) and saves it as
# a .mat file for the matlabs
# Ex. python -O getData.py

from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as sio
#import scipy.signal as sig
#import scipy.constants as const
from astropy.time import Time
import nds2


ifo = 'C1'
# Setup connection to the NDS
ndsServer  = 'nds40.ligo.caltech.edu'
portNumber = 31200
conn       = nds2.connection(ndsServer, portNumber)

# Setup start and stop times
times   = '2017-03-01 00:00:00'
t       = Time(times, format='iso', scale='utc')
t_start = int(t.gps)
dur     = 1000

# channel names
chan_head = ifo + ':' + 'PEM-' + 'RMS' + '_'
sensors   = {'BS'}
dofs      = {'X', 'Y', 'Z'}
bands     = {'0p03_0p1', '0p1_0p3', '0p3_1', '1_3', '3_10', '10_30'}
channels  = []
# why is the channel ordering so weird?
# need to use sorted to preserve the intended ordering
for sensor in sorted(sensors):
    for dof in sorted(dofs):
        for band in sorted(bands):
            channel = chan_head + sensor + '_' + dof + '_' + band
            #print channel
            channels.append(channel)

print("Getting data from " + ndsServer + "...")
data = conn.fetch(t_start, t_start + dur, channels)

if __debug__:
    for i in channels:
        print(i)


# save the data so that it can be loaded by matlab or python
vdata = []
# get the data and stack it into a single matrix where the data are the columns
for k in range(len(channels)):
    vdata.append(data[k].data)

# save to a hdf5 format that matlab can read (why is compression off by default?)
funame = 'data_array.mat'
sio.savemat(funame, mdict={'data': vdata}, do_compression=True)
print("Data saved as " + funame)

if __debug__:
    print("Channel name is " + data[0].channel.name)
    print("Sample rate is " + str(data[0].channel.sample_rate) + " Hz")
    print("Number of samples is " + str(data[0].length))
    print("GPS Start time is " + str(data[0].gps_seconds))

# uncomment this stuff to get info on what fields are in the data
#dir(data[0])
#dir(data[0].channel)
