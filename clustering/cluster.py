"""
cluster.py
Clustering functions. Includes a plotting function as well.

Provides:
compute_kmeans()
cluster_plotter()
"""

from datetime import timedelta
from os import remove
from os.path import abspath, exists

from gwpy.plot import Plot
from gwpy.time import to_gps
from gwpy.timeseries import TimeSeriesDict, TimeSeries
from matplotlib.pyplot import setp
from scipy.stats import zscore
from numpy import stack, concatenate, savetxt
from sklearn.cluster import KMeans
from gwpy.segments import DataQualityFlag

from util import get_logger, Progress, get_path

import h5py

DEFAULT_FILENAME = 'cache.hdf5'

# FIXME A list of colors defines the maximum amount of clusters. Sample from a colormap instead.
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'darkgreen', 'darkblue', 'orangered', 'plum', 'indigo', 'brown',
          'chartreuse', 'pink', 'darkslategray']

# initialize logging.
logger, log_path = get_logger(__name__, verbose=True)
print(f'Writing log to: {log_path}')


def compute_kmeans(channels, start, stop, history=timedelta(hours=2), filename=DEFAULT_FILENAME, downloader=TimeSeriesDict.get, **kwargs):
    """
    Computes k-means clusters and saves the data and labels to filename.
    **kwargs are forwarded to the KMeans constructor.

    >>> from gwpy.time import tconvert, from_gps
    >>> from datetime import timedelta
    >>> from cluster import compute_kmeans
    >>>
    >>> channels = [f'L1:ISI-GND_STS_ETMX_Z_BLRMS_1_3.mean,m-trend', 'L1:ISI-GND_STS_ETMY_Z_BLRMS_1_3.mean,m-trend']
    >>>
    >>> stop = from_gps(60 * (int(tconvert('now')) // 60)) # gets nearest minute to now
    >>> start = stop - timedelta(days=1)  # cluster the past day
    >>> compute_kmeans(channels, start, stop, filename='my_kmeans.hdf5', n_clusters=5, random_state=0)
    """

    # set up duration (minute-trend data has dt=1min, so reject intervals not on the minute).
    duration = (stop - start).total_seconds() / 60
    assert (stop - start).total_seconds() / 60 == (stop - start).total_seconds() // 60
    duration = int(duration)
    logger.info(f'Clustering data from {start} to {stop} ({duration} minutes).')

    # download data using TimeSeries.get(), including history of point at t0.
    logger.debug(f'Initiating download from {start} to {stop} with history={history}...')
    dl = downloader(channels, start=to_gps(start - history), end=to_gps(stop))
    logger.info(f'Downloaded from {start} to {stop} with history={history}.')

    # normalization: compute data stardard score.
    dl_score = TimeSeriesDict()
    dl_score[channel] = [zscore(dl[channel]) for channel in channels]    

    # generate input matrix of the form [sample1;...;sampleN] with sampleK = [feature1,...,featureN]
    # for sklearn.cluster algorithms. This is the slow part of the function, so a progress bar is shown.
    logger.debug(f'Initiating input matrix generation...')
    with Progress('building input', (duration * 60)) as progress:
        input_data = stack(
            [concatenate(
                [progress(dl_score[channel].crop, t,
                          start=to_gps(start + timedelta(seconds=t) - history),
                          end=to_gps(start + timedelta(seconds=t))).value for channel in channels]
            ) for t in range(0, int(duration * 60), 60)])

    # verify input matrix dimensions.
    assert input_data.shape == (duration, int(len(channels) * history.total_seconds() / 60))
    logger.info('Completed input matrix generation.')

    # actually do the fit.
    logger.debug(f'Initiating KMeans({kwargs}) fit...')
    kmeans = KMeans(**kwargs).fit(input_data)
    logger.info(f'Completed KMeans({kwargs}) fit.')

    # write clusters centers to file
    kcenters = kmeans.cluster_centers_    
    clusters = 'cluster_center.txt'
    if exists(clusters):
        remove(clusters)
    savetxt(clusters, kcenters)
    logger.info(f'Wrote clusters center to {clusters}')
    
    # cast the output labels to a TimeSeries so that cropping is easy later on.
    labels = TimeSeries(kmeans.labels_,
                        times=dl[channels[0]].crop(start=to_gps(start), end=to_gps(stop)).times,
                        name='kmeans-labels')

    # put labels in data download dictionary for easy saving.
    dl[labels.name] = labels

    # write data download and labels to specified filename.
    cache_file = abspath(filename)
    if exists(cache_file):
        remove(cache_file)
    dl.write(cache_file)
    logger.info(f'Wrote cache to {filename}')


def cluster_plotter(channels, start, stop,
                    prefix='.',
                    label='kmeans-labels',
                    groups=None,
                    filename=DEFAULT_FILENAME,
                    dqflag='L1:DMT-ANALYSIS_READY:1',
                    xscale=None,
                    unit=None,
                    progressbar=True,
                    **kwargs):
    """
    Plots data with clusters labeled by color in the working directory, or a relative path given by prefix.
    Requires a .hdf5 file produced with a clustering function defined in this module to be in the working directory.
    **kwargs are forwarded to TimeSeries.plot().

    :param prefix: relative path to output images.
    :param label: name attribute of labels TimeSeries saved in filename.
    :param groups: groups of channels to plot in the same figure. See the example.
    :param dqflag: data quality flag for segments bar.
    :param xscale: gps x-axis scale to use.
    :param unit: override y-axis unit.
    :param progressbar: show progress bar.

    >>> from gwpy.time import tconvert, from_gps
    >>> from datetime import timedelta
    >>> from cluster import cluster_plotter
    >>>
    >>> channels = [f'L1:ISI-GND_STS_ETMX_Z_BLRMS_1_3.mean,m-trend', 'L1:ISI-GND_STS_ETMY_Z_BLRMS_1_3.mean,m-trend']
    >>> groups = [[channels, ('ETMX', 'ETMY'), 'L1:ISI-GND_STS_BLRMS_1_3 Z-axis']] # plot on the same figure.
    >>>
    >>> stop = from_gps(60 * (int(tconvert('now')) // 60)) # gets nearest minute to now
    >>> start = stop - timedelta(days=1)  # cluster the past day
    >>> cluster_plotter(channels, start, stop, filename='my_kmeans.hdf5', groups=groups)

    """

    # some defaults.
    if not kwargs:
        kwargs['color'] = 'k'
        kwargs['alpha'] = 0.3
    if groups is None:
        groups = channels

    # read the data from the save file.
    data = TimeSeriesDict.read(filename, channels + [label], start=to_gps(start), end=to_gps(stop))
    logger.info(f'Read {start} to {stop} from {filename}')

    # get segments for the duration specified. Note that this may require doing `ligo-proxy-init -p`.
    logger.debug(f'Getting segments for {dqflag} from {start} to {stop}...')
    dq = DataQualityFlag.query(dqflag, to_gps(start), to_gps(stop))
    logger.info(f'Got segments for {dqflag} from {start} to {stop}.')

    # plotting is slow, so show a nice progress bar.
    logger.debug('Initiating plotting routine...')
    with Progress('plotting', len(channels), quiet=not progressbar) as progress:

        for p, (group, labels, title) in enumerate(groups):

            # plot the group in one figure.
            plt = Plot(*(data[channel] for channel in group), separate = True, sharex = True, zorder = 1, **kwargs)

            # modify the axes one by one.
            axes = plt.get_axes()
            for i, ax in enumerate(axes):

                # namely, add a colored overlay that indicates clustering labels.
                ax.scatter(data[group[i]].times, data[group[i]].value, c=[colors[j] for j in data[label]],
                                  edgecolor='', s=4, zorder=2)

                ax.set_ylabel(f'{labels[i]} {data[group[i]].unit if unit is None else unit}')
                setp(ax.get_xticklabels(), visible=False)

            # modify the figure as a whole.
            plt.add_segments_bar(dq, label='')
            if xscale is not None:
                plt.gca().set_xscale(xscale)
            plt.suptitle(title)

            # save to png.
            progress(plt.save, p, get_path(title, 'png', prefix=prefix))

    logger.info(f'Completed plotting for {start} to {stop} from {filename}')
