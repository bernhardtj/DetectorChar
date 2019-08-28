# Clustering 2019 Library User Guide

## Getting Started
### General conventions
- Make a directory for each set of data or clustering run you perform. Run all code with that as the working directory.
- Clustering minute-trend data and labels is saved in `$(pwd)/cache.hdf5` in GWpy TimeSeries format.
- Postprocessed channels (BLRMS, etc) are saved like
```
$(pwd)/<input_channel_name>\ <repr(processing_start)>.hdf5
```
in GWpy TimeSeries format.
- It also makes logs by module in the current working directory.

### Running Code
To start, do
```sh
mkdir my_cluster_run
cd my_cluster_run
```
Then execute python code within that directory. There are several ways you can do this, either imperatively, via script, or, how I did it, a script of functions that executes commandline arguments:
```python
### tests.py ###

def my_func():
  ...

if __name__ == '__main__':
    from sys import argv
    exec(argv[-1])
```
so you could do e.g. `nohup python -u tests.py 'myfunc()'`.

N.B. This code is for Python 3.7+, so it is recommended to use the following anaconda installation procedure to set up the Python environment.
### Automated Installation of Miniconda 3.7
1. Copy `etc/Makefile` in this repository to a new directory `PREFIX` of your choice.
2. Also copy `etc/PKGS` to `$PREFIX`, or make a similar `PKGS` file. 
3. Run `make install`, which will provision a Miniconda python environment with the conda packages in `PKGS` in `$PREFIX/env`.

Python will now be accessible at `$PREFIX/env/bin/python`.


## `channeling.py`: BLRMS generation, minute trend, etc.
### Running
1. Set up and change to a new directory for processing NDS-downloaded channels.
2. Make a `config.ini` in this directory. An example is included with this repo in `etc`.
3. Run something like:
```python
from channeling import PostProcessor, process_channel, config
from util import config2dataclass
for channel in eval(config['DEFAULT']['channels']):
    for processor in config2dataclass(PostProcessor, config, channel): # returns list of processors assigned to the channel
    process_channel(processor, start, stop)
```
or, alternatively,
```python
from channeling import process_channel, MinuteTrend
from datetime import timedelta
process_channel(MinuteTrend(channel='L1:GDS-CALIB_STRAIN',
                            stride_length=timedelta(weeks=1),
                            respect_segments=False,
                            postprocessor=[], bands=[], channels=[]),
                start, stop)
 ```
 where `start` and `stop` are `datetime.datetime` objects.
 
 The function `config2dataclass` returns instantiated `PostProcessor` child-classes where key/value pairs in relevant config sections are passed as construction arguments.
 The second example passes these paramters manually.
 
 ### `config.ini` Syntax
 Keys can be specified in the `DEFAULT` section, or in sections named by channel or by `PostProcessor` (`BLRMS`, `MinuteTrend`, etc).
 Required keys are:
 - `channels`
 - `bands`
 - `PostProcessor`
 - `stride_length`
 - `respect_segments`
 
 Note that INI is case-insensitive. The `respect_segments` option sets non-observing segments to zero.
 
 ### Reading data out of `channeling.py` save files
 Some functions, including `process_channel` and `compute_kmeans` allow you to set an alternative to `TimeSeriesDict.get` for data downloading via the `downloader` keyword argument.
 The function `channeling.channeling_reader(input_channels, start_of_generating, search_dirs=['.'])` returns such an alternative.
 It searches for HDF5 files named with a channel in the `input_channels` and the `start_of_generating` in one of the `search_dirs`.
 
 ## `cluster.py`: Clustering
 The `compute_kmeans` and `cluster_plotter` functions are of interest here. For example:
 ```python
 from gwpy.time import tconvert, from_gps
 from datetime import timedelta
 from cluster import compute_kmeans
 channels = [f'L1:ISI-GND_STS_ETMX_Z_BLRMS_1_3.mean,m-trend', 'L1:ISI-GND_STS_ETMY_Z_BLRMS_1_3.mean,m-trend']
 stop = from_gps(60 * (int(tconvert('now')) // 60)) # gets nearest minute to now
 start = stop - timedelta(days=1)  # cluster the past day
 # cluster
 compute_kmeans(channels, start, stop, n_clusters=5, random_state=0)
 # plot time series
 groups = [[channels, ('ETMX', 'ETMY'), 'L1:ISI-GND_STS_BLRMS_1_3 Z-axis']] # plot on the same figure.
 cluster_plotter(channels, start, stop, groups=groups)
 ```
 
 ## `attributer.py`: Cluster Evaluation
 Two functions are defined here. This file is not commented and cleaned up as well as the other files in this package, due to time constraints.
 ### Representative Spectra
 ```python
 from attributer import representative_spectra
 representative_spectra(['L1:GDS-CALIB_STRAIN'], start, stop, 4096) # resampling to 4096 Hz before saving raw data
 ```
 NOTE: The `representative_spectra` function uses the `groups` argument for plotting. Currently no groups is broken, so the above code won't work exactly.
 The work-around could be `groups=[[[o], '_', o] for o in ['L1:GDS-CALIB_STRAIN']]`.
 ### Percent Increase Tables
 ```python
 from attributer import threshold_table
 bands = [(100,200)]
 threshold_table(start, stop, ['L1:GDS-CALIB_STRAIN_BLRMS_100_200.mean,m-trend'], ['L1:GDS-CALIB_STRAIN'], [f'{a}_{b}' for a, b in bands])
 ```
 The first channel list argument is the list of minute-trend BLRMS channels to read out of `$(pwd)/cache.hdf5`. The second channel list argument is a list of prefixes without the `_BLRMS_X_Y.mean,m-trend`.
 
 ## Final Notes
 Many example invocations can be found in `tests.py`, in the hairy mess of commented stuff.
 
 It is still recommended to make your own method of running the functions in this library.
