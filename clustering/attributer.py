"""
attributer.py
get some cluster attributes.

"""

from gwpy.frequencyseries import FrequencySeries
from gwpy.plot import Plot
from gwpy.time import to_gps
from gwpy.timeseries import TimeSeriesDict, TimeSeries
from ligotimegps import LIGOTimeGPS
from numpy import stack, median, diff

from cluster import DEFAULT_FILENAME, colors
from util import get_logger, get_path, write_to_disk, better_aa_opts, data_exists, Progress, writing_opts, path2h5file

# initialize logging.
logger, log_path = get_logger(__name__, verbose=True)
print(f'Writing log to: {log_path}')


def threshold_table(start, stop, reading_channels, channels, bands, label='kmeans-labels', filename=DEFAULT_FILENAME,
                    prefix='.'):
    """
    Makes a html table of 'percent increase' from the largest cluster by band and channel.
    """
    data = TimeSeriesDict.read(filename, reading_channels + [label], start=to_gps(start), end=to_gps(stop))
    labels = data[label]

    clusters = list(range(max(labels.value) + 1))
    cluster_counts = list(len(labels.value[labels.value == c]) for c in clusters)
    largest_cluster = cluster_counts.index(max(cluster_counts))
    clusters.remove(largest_cluster)

    logger.info(
        f'Largest cluster found to be Nº{largest_cluster} ({100 * max(cluster_counts) // len(labels.value)}%). Doing {clusters}.')
    cluster_counts.remove(max(cluster_counts))

    def amplitude(channel, cluster):
        """return median amplitude for channel in cluster."""
        try:
            chan = data[channel]
        except KeyError:
            return 0.0
        return median([chan.value[i] for i, c in enumerate(labels.value) if c == cluster])

    def threshold(cluster, channel, band) -> str:
        f_channel = f'{channel}_BLRMS_{band}.mean'
        base = amplitude(f_channel, largest_cluster)
        if base != 0.0:
            return str(int(100 * (amplitude(f_channel, cluster) - base) / base)) + '%'
        else:
            return str(amplitude(f_channel, cluster))

    range_chan = 'L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'
    if range_chan in reading_channels:
        base_range = amplitude(range_chan, largest_cluster)
        if base_range != 0.0:
            snsh = lambda c: 'SNSH: ' + str(int(100 * (amplitude(range_chan, c) - base_range) / base_range)) + '%'
        else:
            snsh = lambda c: 'SNSH: 0.0'
    else:
        snsh = lambda c: ''

    with Progress('taking thresholds', len(clusters)) as progress:
        for i, cluster in enumerate(clusters):
            buffer = [[''] + bands]
            for channel in channels:
                buffer.append([channel] + [progress(threshold, i, cluster, channel, band) for band in bands])
            html_table(f'cluster {cluster} ({colors[cluster]}) {snsh(cluster)}',
                       csv_writer(buffer, get_path(f'{cluster}', 'csv', prefix=prefix)),
                       get_path(f'{cluster}', 'html', prefix=prefix))
    html_table('Index', csv_writer(
        [['clusters:']] + [[f'<a href="{cluster}.html">Nº{cluster} ({colors[cluster]})</a>'] for cluster in clusters],
        get_path('idx', 'csv', prefix=prefix)), get_path('index', 'html', prefix=prefix))


def representative_spectra(channels, start, stop, rate, label='kmeans-labels', filename=DEFAULT_FILENAME, prefix='.',
                           downloader=TimeSeriesDict.get, cluster_numbers=None, groups=None, **kwargs):
    """
    Make representative spectra for each cluster based on the median psd for minutes in that cluster.
    Downloads only the raw minutes in the cluster to save.
    """
    if groups is None:
        groups = channels

    # read the labels from the save file.
    labels = TimeSeries.read(filename, label, start=to_gps(start), end=to_gps(stop))
    logger.info(f'Read labels {start} to {stop} from {filename}')

    if cluster_numbers is None:
        clusters = list(range(max(labels.value) + 1))

        cluster_counts = list(len(labels.value[labels.value == c]) for c in clusters)
        largest_cluster = cluster_counts.index(max(cluster_counts))
        clusters.remove(largest_cluster)

        logger.info(
            f'Largest cluster found to be Nº{largest_cluster} ({100 * max(cluster_counts) // len(labels.value)}%). Doing {clusters}.')
        cluster_counts.remove(max(cluster_counts))
    else:
        clusters = cluster_numbers
        cluster_counts = list(len(labels.value[labels.value == c]) for c in clusters)

    t, v, d = labels.times, labels.value, diff(labels.value)

    pairs = list(zip([t[0]] + list(t[:-1][d != 0]), list(t[1:][d != 0]) + [t[-1]]))
    values = list(v[:-1][d != 0]) + [v[-1]]
    assert len(pairs) == len(values)  # need to include start-| and |-end
    # l|r l|r l|r l|r
    # l,r l,r l,r l,r
    # l r,l r,l r,l r # zip(start + l[1:], r[:-1] + stop)

    print(pairs)
    for pair in pairs:
        print(int(pair[1].value) - int(pair[0].value))
    print(values)

    # use h5py to make a mutable object pointing to a file on disk.
    save_file, filename = path2h5file(get_path(f'spectra-cache {start}', 'hdf5', prefix=prefix))
    logger.debug(f'Initiated hdf5 stream to {filename}')

    logger.info(f'Patching {filename}...')
    for i, (dl_start, end) in enumerate(pairs):
        if values[i] in clusters:
            if not data_exists(channels, to_gps(end).seconds, save_file):
                logger.debug(f'Downloading Nº{values[i]} from {dl_start} to {end}...')
                try:
                    dl = downloader(channels, start=to_gps(dl_start) - LIGOTimeGPS(60),
                                    end=to_gps(end) + LIGOTimeGPS(seconds=1))
                    out = TimeSeriesDict()
                    for n in dl:
                        out[n] = dl[n].resample(**better_aa_opts(dl[n], rate))
                    write_to_disk(out, to_gps(dl_start).seconds, save_file)
                except RuntimeError:  # Cannot find all relevant data on any known server
                    logger.warning(f"SKIPPING Nº{values[i]} from {dl_start} to {end} !!")

    logger.info('Reading data...')
    data = TimeSeriesDict.read(save_file, channels)

    logger.info('Starting PSD generation...')

    f = data[channels[0]].crop(start=to_gps(data[channels[0]].times[-1]) - LIGOTimeGPS(60),
                               end=to_gps(data[channels[0]].times[-1])).psd().frequencies

    d = (to_gps(labels.times[-1]).seconds - to_gps(labels.times[1]).seconds)
    for i, cluster in enumerate(clusters):
        try:
            psds = {channel: FrequencySeries.read(filename, f'{cluster}-{channel}') for channel in channels}
            logger.info(f'Loaded Nº{cluster}.')

        except KeyError:

            logger.info(f'Doing Nº{cluster} ({100 * cluster_counts[i] / len(labels.value):.2f}% of data)...')
            with Progress(f'psd Nº{cluster} ({i + 1}/{len(clusters)})', len(channels) * d) as progress:
                psds = {channel: FrequencySeries(median(stack([progress(data[channel].crop,
                                                                        pc * d + (to_gps(time).seconds - to_gps(
                                                                            labels.times[1]).seconds),
                                                                        start=to_gps(time) - LIGOTimeGPS(60),
                                                                        end=to_gps(time)).psd().value
                                                               for c, time in zip(labels.value, labels.times) if
                                                               c == cluster]),
                                                        axis=0), frequencies=f, name=f'{cluster}-{channel}')
                        for pc, channel in enumerate(channels)}
            for name in psds.keys():
                psds[name].write(filename, **writing_opts)

        # plotting is slow, so show a nice progress bar.
        logger.debug('Initiating plotting routine...')
        with Progress('plotting', len(groups)) as progress:

            for p, (group, lbls, title) in enumerate(groups):
                # plot the group in one figure.
                plt = Plot(*(psds[channel] for channel in group), separate=False, sharex=True, zorder=1, **kwargs)
                # plt.gca().set_xlim((30,60))
                # modify the figure as a whole.
                # plt.add_segments_bar(dq, label='')
                plt.gca().set_xscale('log')
                plt.gca().set_yscale('log')
                plt.suptitle(title)
                plt.legend(lbls)

                # save to png.
                progress(plt.save, p, get_path(f'{cluster}-{title}', 'png', prefix=f'{prefix}/{cluster}'))


def csv_writer(buffer, filename, delimiter=','):
    with open(filename, "w+") as f:
        f.writelines([delimiter.join(line) + '\n' for line in buffer])
    return filename


def html_table(title, in_filename, out_filename):
    filein = open(in_filename, "r")
    fileout = open(out_filename, "w+")
    data = filein.readlines()

    table = f"<!doctype html><html lang='en'><head><title>{title}</title></head><body><h1>{title}</h1><table>"

    # Create the table's column headers
    header = data[0].split(",")
    table += "  <tr>\n"
    for column in header:
        table += "    <th>{0}</th>\n".format(column.strip())
    table += "  </tr>\n"

    # Create the table's row data
    for line in data[1:]:
        row = line.split(",")
        table += "  <tr>\n"
        for column in row:
            table += "    <td>{0}</td>\n".format(column.strip())
        table += "  </tr>\n"

    table += "</table></body></html>"

    fileout.writelines(table)
    fileout.close()
    filein.close()
