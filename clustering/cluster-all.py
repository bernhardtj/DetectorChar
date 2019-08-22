"""
cluster-all.py

Hacky paste of https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
to quickly look at other algorithms. Intermediate products are cached as .npy files.
"""

import warnings
from datetime import timedelta
from os import remove
from os.path import abspath, exists

import numpy as np
from gwpy.time import to_gps
from gwpy.timeseries import TimeSeriesDict, TimeSeries
from numpy import stack, concatenate
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from util import get_logger, Progress

DEFAULT_FILENAME = 'cache.hdf5'

# initialize logging.
logger, log_path = get_logger(__name__, verbose=True)
print(f'Writing log to: {log_path}')


def compute_all(channels, start, stop, history=timedelta(hours=2), filename=DEFAULT_FILENAME, **kwargs):
    # set up duration (minute-trend data has dt=1min, so reject intervals not on the minute).
    duration = (stop - start).total_seconds() / 60
    assert (stop - start).total_seconds() / 60 == (stop - start).total_seconds() // 60
    duration = int(duration)
    logger.info(f'Clustering data from {start} to {stop} ({duration} minutes).')

    # download data using TimeSeries.get(), including history of point at t0.
    logger.debug(f'Initiating download from {start} to {stop} with history={history}...')
    dl = TimeSeriesDict.get(channels, start=to_gps(start - history), end=to_gps(stop))
    logger.info(f'Downloaded from {start} to {stop} with history={history}.')

    if exists('input.npy'):
        input_data = np.load('input.npy')
        logger.info('Loaded input matrix.')
    else:
        # generate input matrix of the form [sample1;...;sampleN] with sampleK = [feature1,...,featureN]
        # for sklearn.cluster algorithms. This is the slow part of the function, so a progress bar is shown.
        logger.debug(f'Initiating input matrix generation...')
        with Progress('building input', (duration * 60)) as progress:
            input_data = stack(
                [concatenate(
                    [progress(dl[channel].crop, t,
                              start=to_gps(start + timedelta(seconds=t) - history),
                              end=to_gps(start + timedelta(seconds=t))).value for channel in channels]
                ) for t in range(0, int(duration * 60), 60)])

        # verify input matrix dimensions.
        assert input_data.shape == (duration, int(len(channels) * history.total_seconds() / 60))
        np.save('input.npy', input_data)
        logger.info('Completed input matrix generation.')

    params = {'quantile': .3,
              'eps': .3,
              'damping': .9,
              'preference': -200,
              'n_neighbors': 10,
              'n_clusters': 15,
              'min_samples': 20,
              'xi': 0.05,
              'min_cluster_size': 0.1}

    if exists('X.npy'):
        X = np.load('X.npy')
        logger.info('Loaded X')
    else:
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(input_data)
        np.save('X.npy', X)
        logger.info('Generated X')

    if exists('bandwidth.npy'):
        bandwidth = np.load('bandwidth.npy')
        logger.info('Loaded bandwidth')
    else:
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
        np.save('bandwidth.npy', bandwidth)
        logger.info('Generated bandwidth')

    if exists('connectivity.npy'):
        connectivity = np.load('connectivity.npy', allow_pickle=True)
        logger.info('Loaded connectivity')
    else:
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        np.save('connectivity.npy', connectivity)
        logger.info('Generated connectivity')

    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    optics = cluster.OPTICS(min_samples=params['min_samples'],
                            xi=params['xi'],
                            min_cluster_size=params['min_cluster_size'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('DBSCAN', dbscan),
        ('OPTICS', optics),
        ('Birch', birch),
        ('GaussianMixture', gmm)
        # ('Ward', ward),
        # ('AgglomerativeClustering', average_linkage),
    )

    for name, algorithm in clustering_algorithms:
        if exists(f'part-{name}-{filename}'):
            labels = TimeSeries.read(f'part-{name}-{filename}', f'{name}-labels')
            logger.debug(f'LOADED {name}.')
        else:
            logger.debug(f'doing {name}...')
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                            "connectivity matrix is [0-9]{1,2}" +
                            " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                            " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)

            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)
            # cast the output labels to a TimeSeries so that cropping is easy later on.
            labels = TimeSeries(y_pred,
                                times=dl[channels[0]].crop(start=to_gps(start), end=to_gps(stop)).times,
                                name=f'{name}-labels')

            labels.write(f'part-{name}-{filename}')
        # put labels in data download dictionary for easy saving.
        dl[labels.name] = labels

    # write data download and labels to specified filename.
    cache_file = abspath(filename)
    if exists(cache_file):
        remove(cache_file)
    dl.write(cache_file)
    logger.info(f'Wrote cache to {filename}')
