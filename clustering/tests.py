"""
tests.py
Various runnable functions from SURF 2019.
Not terribly useful.
"""


from datetime import datetime, timedelta

start = datetime(year=2019, month=5, day=31, hour=23, minute=59, second=42)
stop = start + timedelta(minutes=60 * 24 * 30)  # from_gps(60 * (int(tconvert('now')) // 60))  # nearest minute
print(f'tests.py: defined start={start}, stop={stop}')


def blrms_generation(channel):
    from channeling import config, process_channel, PostProcessor
    from util import config2dataclass

    return [process_channel(processor, start, stop + timedelta(weeks=1)) for processor in
            config2dataclass(PostProcessor, config, channel)]


def reprocess_darm(channel):
    from channeling import config, process_channel, PostProcessor, channeling_reader
    from util import config2dataclass

    return [process_channel(processor, start, stop + timedelta(weeks=1),
                            downloader=channeling_reader([channel, 'L1:GDS-CALIB_STRAIN'], start)) for processor in
            config2dataclass(PostProcessor, config, channel)]


def multi_channeling(func):
    from multiprocessing import Pool
    from channeling import config

    pool = Pool()

    pool.map(func, eval(config['DEFAULT']['channels']))


def test_kmeans_sei():
    from datetime import timedelta
    from cluster import cluster_plotter, compute_kmeans
    from channeling import channeling_reader
    from attributer import threshold_table, representative_spectra

    channels = [f'L1:ISI-GND_STS_{sensor}_{dof}_BLRMS_{band}.mean'
                for dof in ['X', 'Y', 'Z']
                for band in ['30M_100M', '100M_300M', '300M_1', '1_3', '3_10', '10_30']
                for sensor in ['ETMX', 'ETMY', 'ITMY']]

    in_channels = [f'{c},m-trend' for c in channels]

    in_darm = [f'L1:GDS-CALIB_STRAIN_BLRMS_{band}' for band in [
        '10_13', '18_22', '22_27', '27_29', '29_40', '40_54', '54_65', '65_76', '75_115', '115_190', '190_210',
        '210_290', '290_480', '526_590', '590_650', '650_885', '885_970', '1110_1430']]

    out_darm = [f'{c}.mean' for c in in_darm]

    # L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean # is already minute-trend

    # compute_kmeans(channels, start + timedelta(hours=2), stop, history=timedelta(hours=2),
    #                 downloader=channeling_reader(in_channels + in_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], start, search_dirs=['../../../blrms/sei', '../../../blrms/darm']),
    #                n_clusters=3, random_state=0)
    #
    # cluster_plotter(channels, start + timedelta(hours=2), stop,
    #                 groups=[
    #                            [[f'L1:ISI-GND_STS_{sensor}_{dof}_BLRMS_{band}.mean' for dof in 'XYZ'], 'XYZ',
    #                             f'L1:ISI-GND_STS_{sensor}_BLRMS_{band}.mean'] for band in ['1_3', '3_10'] for sensor in
    #                            #['30M_100M', '100M_300M', '300M_1', '1_3', '3_10', '10_30'] for sensor in
    #                            ['ETMX', 'ETMY']],# 'ITMY']],# + [[[d], '1', d] for d in
    #                                                     #    out_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean']],
    #                 figsize=(100, 10), xscale='days', unit='[nm/s]')

    in_channels = [f'L1:ISI-GND_STS_{sensor}_{dof}'
                for dof in ['X', 'Y', 'Z']
                for sensor in ['ETMX', 'ETMY', 'ITMY']]

    threshold_table(start + timedelta(hours=2), stop, channels + out_darm, in_channels + ['L1:GDS-CALIB_STRAIN'], ['30M_100M', '100M_300M', '300M_1', '1_3', '3_10', '10_30'] + [
        '10_13', '18_22', '22_27', '27_29', '29_40', '40_54', '54_65', '65_76', '75_115', '115_190', '190_210',
        '210_290', '290_480', '526_590', '590_650', '650_885', '885_970', '1110_1430'], prefix='thresh')



def make_minute_trend(chans):
    from channeling import channeling_reader, process_channel, MinuteTrend
    channel, out_channels = chans
    return process_channel(MinuteTrend(channels=[], channel=channel, stride_length=timedelta(weeks=1),
                                    respect_segments=False, postprocessor=['MinuteTrend'], bands=[]),
                        start, stop + timedelta(weeks=1),
                        downloader=channeling_reader(out_channels, start))

def test_kmeans_mic():
    from cluster import compute_kmeans, cluster_plotter
    from channeling import channeling_reader, process_channel, MinuteTrend
    from attributer import threshold_table, representative_spectra
    from gwpy.segments import DataQualityFlag

    in_channels = [  # "L1:PEM-CS_MIC_EBAY_RACKS_DQ",
        "L1:PEM-CS_MIC_LVEA_BS_DQ",
        # "L1:PEM-CS_MIC_LVEA_INPUTOPTICS_DQ",
        # "L1:PEM-CS_MIC_LVEA_OUTPUTOPTICS_DQ",
        # "L1:PEM-CS_MIC_LVEA_XMANSPOOL_DQ",
        # "L1:PEM-CS_MIC_LVEA_YMANSPOOL_DQ",
        # "L1:PEM-CS_MIC_PSL_CENTER_DQ",
        # "L1:PEM-EX_MIC_EBAY_RACKS_DQ",
        "L1:PEM-EX_MIC_VEA_PLUSX_DQ",
        # "L1:PEM-EY_MIC_EBAY_RACKS_DQ",
        "L1:PEM-EY_MIC_VEA_PLUSY_DQ",
        "L1:PEM-MX_MIC_VEA_MIDX_DQ",
        "L1:PEM-MY_MIC_VEA_MIDY_DQ"]
    bands = [(10, 28),
             (28, 32),
             (32, 50),
             (50, 70),
             (70, 100),
             (100, 200)]
    out_channels = [f'{c}_BLRMS_{a}_{b}' for a, b in bands for c in in_channels]
    outer_channels = [f'{c}.mean' for c in out_channels]

    in_darm = [f'L1:GDS-CALIB_STRAIN_BLRMS_{band}' for band in [
        '10_13', '18_22', '22_27', '27_29', '29_40', '40_54', '54_65', '65_76', '75_115', '115_190', '190_210',
        '210_290', '290_480', '526_590', '590_650', '650_885', '885_970', '1110_1430']]

    out_darm = [f'{c}.mean' for c in in_darm]

    # from multiprocessing import Pool
    # pool = Pool()
    # pool.map(make_minute_trend, ([o, in_channels] for o in out_channels))

    # compute_kmeans(outer_channels + out_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], start + timedelta(hours=2), stop, history=timedelta(hours=2),
    #                downloader=channeling_reader(out_channels + in_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], start, search_dirs=['../../blrms/mic', '../../blrms/darm']),
    #                n_clusters=15, random_state=0)

    # cluster_plotter(outer_channels + out_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], start + timedelta(hours=2), stop, figsize=(100, 10), xscale='days',
    #                 groups=[[[f'{c}_BLRMS_{a}_{b}.mean' for c in in_channels], 'LXYMN', f'MIC_BLRMS_{a}_{b}'] for a, b in bands] + [[['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], ' ', 'L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean']])
    #
    # threshold_table(start + timedelta(hours=2), stop, outer_channels + out_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], in_channels + ['L1:GDS-CALIB_STRAIN'], [f'{a}_{b}' for a, b in bands] + [
    #     '10_13', '18_22', '22_27', '27_29', '29_40', '40_54', '54_65', '65_76', '75_115', '115_190', '190_210',
    #     '210_290', '290_480', '526_590', '590_650', '650_885', '885_970', '1110_1430'], prefix='thresh')

    # blue 2
    # darkblue 7
    # m 5

    # representative_spectra(in_channels, start + timedelta(days=7), start+timedelta(days=11), 512, cluster_numbers=[2], prefix='spectra', groups=[[[o], '1', o] for o in in_channels], figsize=(18, 10))
    # representative_spectra(in_channels, (1244270700//60)*60, (1244313540//60)*60, 512, cluster_numbers=[5], prefix='spectra', groups=[[[o], '1', o] for o in in_channels], figsize=(18, 10))
    # representative_spectra(in_channels, start + timedelta(days=7), 1244313601, 512, prefix='spectra', cluster_numbers=[2], groups=[[[o], '1', o] for o in in_channels], figsize=(18, 10))
    # representative_spectra(['L1:GDS-CALIB_STRAIN'],(1244270700//60)*60, (1244313540//60)*60, 4096, cluster_numbers=[5], prefix='spectra', groups=[[[o], '1', o] for o in ['L1:GDS-CALIB_STRAIN']], figsize=(18, 10))
    # representative_spectra(['L1:GDS-CALIB_STRAIN'],start + timedelta(days=7), start+timedelta(days=11), 4096, cluster_numbers=[2], prefix='spectra', groups=[[[o], '1', o] for o in ['L1:GDS-CALIB_STRAIN']], figsize=(18, 10))
    # for c in in_channels:
    #      plot_spectra([5, 2, 7], c, legend=['quiet',  'longer', 'burst' ], xlim=(10,100))

    # representative_spectra(in_channels, start + timedelta(days=4), start+ timedelta(days=6), 512, prefix='spectra', cluster_numbers=[7], groups=[[[o], '1', o] for o in in_channels], figsize=(18, 10))

    # print([[int(s.start), int(s.end)] for s in
    #                 DataQualityFlag.query('L1:DMT-ANALYSIS_READY:1', 1244346000, 1244391840).active
    #                 ])
    #
    # representative_spectra(['L1:GDS-CALIB_STRAIN'],1244346000, 1244391840, 4096, cluster_numbers=[2,5], prefix='spectra', groups=[[[o], '1', o] for o in ['L1:GDS-CALIB_STRAIN']], figsize=(18, 10))
    for c in ['L1:GDS-CALIB_STRAIN']:
        plot_spectra([5,2], c, legend=['quiet', 'longer'], xlim=(10,50), unit='strain')

def test_kmeans_acc():
    from cluster import compute_kmeans, cluster_plotter
    from channeling import channeling_reader
    from attributer import representative_spectra, threshold_table
    from gwpy.segments import DataQualityFlag

    # in_channels = ["L1:PEM-CS_ACC_LVEAFLOOR_HAM1_X_DQ",
    #                "L1:PEM-CS_ACC_LVEAFLOOR_HAM1_Y_DQ",
    #                "L1:PEM-CS_ACC_LVEAFLOOR_HAM1_Z_DQ",
    #                "L1:PEM-MX_ACC_BEAMTUBE_VEA_X_DQ",
    #                "L1:PEM-MY_ACC_BEAMTUBE_VEA_Y_DQ",
    #                "L1:PEM-EX_ACC_BSC4_ETMX_X_DQ",
    #                "L1:PEM-EX_ACC_BSC4_ETMX_Y_DQ",
    #                "L1:PEM-EX_ACC_BSC4_ETMX_Z_DQ",
    #                "L1:PEM-EY_ACC_BSC5_ETMY_X_DQ",
    #                "L1:PEM-EY_ACC_BSC5_ETMY_Y_DQ",
    #                "L1:PEM-EY_ACC_BSC5_ETMY_Z_DQ"]
 
    in_channels = ["L1:PEM-CS_ACC_BEAMTUBE_XMAN_Y_DQ",
                   "L1:PEM-CS_ACC_BEAMTUBE_YMAN_X_DQ",
                   "L1:PEM-EX_ACC_BEAMTUBE_MAN_X_DQ",
                   "L1:PEM-EX_ACC_BEAMTUBE_MAN_Z_DQ",
                   "L1:PEM-EY_ACC_BEAMTUBE_MAN_Y_DQ",
                   "L1:PEM-EY_ACC_BEAMTUBE_MAN_Z_DQ",
                   "L1:PEM-MX_ACC_BEAMTUBE_2100X_X_DQ",
                   "L1:PEM-MX_ACC_BEAMTUBE_VEA_Z_DQ",
                   "L1:PEM-MY_ACC_BEAMTUBE_2100Y_Y_DQ",
                   "L1:PEM-MY_ACC_BEAMTUBE_VEA_Z_DQ"]

    bands = [(1, 4),
             (4, 10),
             (10, 28),
             (28, 32),
             (32, 48),
             (48, 60),
             (60, 80),
             (80, 118),
             (118, 122)]

    out_channels = [f'{c}_BLRMS_{a}_{b}' for a, b in bands for c in in_channels]
    outer_channels = [f'{c}.mean' for c in out_channels]

    in_darm = [f'L1:GDS-CALIB_STRAIN_BLRMS_{band}' for band in [
        '10_13', '18_22', '22_27', '27_29', '29_40', '40_54', '54_65', '65_76', '75_115', '115_190', '190_210',
        '210_290', '290_480', '526_590', '590_650', '650_885', '885_970', '1110_1430']]

    out_darm = [f'{c}.mean' for c in in_darm]


    # from multiprocessing import Pool
    # pool = Pool(4)
    # pool.map(make_minute_trend, ([o, in_channels] for o in out_channels))

    # compute_kmeans(outer_channels + out_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], start + timedelta(hours=2), stop, history=timedelta(hours=2),
    #                downloader=channeling_reader(out_channels + in_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], start, search_dirs=['../../blrms/accel', '../../blrms/darm']),
    #                n_clusters=15, random_state=0)
    #
    # cluster_plotter(outer_channels + out_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], start + timedelta(hours=2), stop, figsize=(100, 10), xscale='days', groups=[[[o], '1', o] for o in outer_channels + out_darm + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean']])


    # threshold_table(start + timedelta(hours=2), stop, outer_channels, in_channels, [f'{a}_{b}' for a, b in bands], prefix='thresh')
    # threshold_table(start + timedelta(hours=2), stop, outer_channels + out_darm  + ['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], in_channels + ['L1:GDS-CALIB_STRAIN'], [f'{a}_{b}' for a, b in bands] + [
    #     '10_13', '18_22', '22_27', '27_29', '29_40', '40_54', '54_65', '65_76', '75_115', '115_190', '190_210',
    #     '210_290', '290_480', '526_590', '590_650', '650_885', '885_970', '1110_1430'], prefix='thresh')
    # representative_spectra(in_channels, 1244651160, 1244691600, 256, cluster_numbers=[0,4], prefix='spectra', groups=[[[o], '1', o] for o in in_channels], figsize=(18, 10))
    # print([[int(s.start), int(s.end)] for s in
    #                 DataQualityFlag.query('L1:DMT-ANALYSIS_READY:1', 1243975560, 1243997520).active
    #                 ])
    # representative_spectra(['L1:GDS-CALIB_STRAIN'], 1243975560, 1243997520, 4096, cluster_numbers=[0, 4], prefix='spectra',
    #                         groups=[[[o], '1', o] for o in ['L1:GDS-CALIB_STRAIN']], figsize=(18, 10))

    for c in ['L1:GDS-CALIB_STRAIN']:
        plot_spectra([4,0] , c, legend=['Quiet', 'Fans'], xlim=(10,100), unit='strain')

    # for c in in_channels:
    #     plot_spectra([4, 0], c, legend=['Quiet', 'Fans'], xlog=False, xlim=(55, 65))

def make_psds():
    from attributer import representative_spectra
    channels = [f'L1:ISI-GND_STS_{sensor}_{dof}_DQ' for dof in ['X', 'Y', 'Z'] for sensor in ['ETMX', 'ETMY', 'ITMY']]
    # groups = [
    #     [[f'L1:ISI-GND_STS_{sensor}_{dof}_DQ' for dof in 'XYZ'], 'XYZ',
    #      f'L1:ISI-GND_STS_{sensor}_DQ'] for sensor in ['ETMX', 'ETMY', 'ITMY']]
    # representative_spectra(channels, start, stop, prefix='spectra', groups=groups, figsize=(18, 10))

    groups = [
        [[f'L1:ISI-GND_STS_{sensor}_{dof}_DQ' for sensor in ['ETMX', 'ETMY', 'ITMY']], ['ETMX', 'ETMY', 'ITMY'],
         f'L1:ISI-GND_STS_{dof}_DQ'] for dof in 'XYZ']

    representative_spectra(channels, start, stop, prefix='spectra', groups=groups, figsize=(18, 10))


def overview_figs():
    from cluster import cluster_plotter

    channels = [f'L1:ISI-GND_STS_{sensor}_{dof}_BLRMS_{band}.mean,m-trend'
                for dof in ['X', 'Y', 'Z']
                for band in ['30M_100M', '100M_300M', '300M_1', '1_3', '3_10', '10_30']
                for sensor in ['ETMX', 'ETMY', 'ITMY']]

    cluster_plotter(channels, start, stop, groups=[
        [[f'L1:ISI-GND_STS_{sensor}_{dof}_BLRMS_{band}.mean,m-trend' for dof in 'XYZ'], 'XYZ',
         f'L1:ISI-GND_STS_{sensor}_BLRMS_{band}.mean,m-trend'] for band in
        ['30M_100M', '100M_300M', '300M_1', '1_3', '3_10', '10_30'] for sensor in ['ETMX', 'ETMY', 'ITMY']],
                    figsize=(100, 10), xscale='days', unit='[nm/s]')


def report1_figs():
    from datetime import timedelta
    from cluster import cluster_plotter

    prefix = 'report_final'
    sensor = 'ETMY'

    # (start-day, duration-in-days)
    figs = dict(#earthquakes={'30M_100M': [(16, 1)]},#((2, 0.75) (24, 0.75), (14.75, 0.5), (2, 0.75)]},
                 #wind={},
                # microseism={'100M_300M': [(8, 4)]},#(4, 4), , (22, 4)]},
                 daynight={'1_3': [(2, 3), (12, 9)]},
                # trains={'1_3': [(8, 3)]}
        #trains = {'1_3': [(8,3)]}
                )

    # TRY 3 days from 20 for the trains pic.

    for event in figs.keys():                        #         groups=[
                        # [[f'L1:ISI-GND_STS_{sensor}_{dof}_BLRMS_{band}.mean,m-trend' for dof in 'XYZ'], 'XYZ',
                        #  f'{event}-day{day}_{days}-L1:ISI-GND_STS_{sensor}_BLRMS_{band}.mean,m-trend']],
        for band in figs[event].keys():
            for day, days in figs[event][band]:
                cluster_plotter([f'L1:ISI-GND_STS_{sensor}_{dof}_BLRMS_{band}.mean,m-trend' for dof in 'Z'],
                                start + timedelta(days=day),
                                start + timedelta(days=day) + timedelta(days=days), groups=[
                        [[f'L1:ISI-GND_STS_{sensor}_{dof}_BLRMS_{band}.mean,m-trend' for dof in 'Z'], 'Z',
                         f'{event}-day{day}_{days}-L1:ISI-GND_STS_{sensor}_BLRMS_{band}.mean,m-trend']],
                                prefix=prefix, progressbar=False, unit='[nm/s]', figsize=(9,3))


def plot_blrms():
    from channeling import channeling_reader
    from gwpy.segments import DataQualityFlag
    from gwpy.time import to_gps
    in_darm = [f'L1:GDS-CALIB_STRAIN_BLRMS_{band}' for band in [
        '10_13', '18_22', '22_27', '27_29', '29_40', '40_54', '54_65', '65_76', '75_115', '115_190', '190_210',
        '210_290', '290_480', '526_590', '590_650', '650_885', '885_970', '1110_1430']]
    stop = start + timedelta(days=1)
    out_darm = [f'{c}.mean' for c in in_darm]
    dqflag = 'L1:DMT-ANALYSIS_READY:1'
    dq = DataQualityFlag.query(dqflag, to_gps(start), to_gps(stop))
    downloader = channeling_reader(['L1:GDS-CALIB_STRAIN'], start, search_dirs=['../darm'])
    data = downloader(in_darm, start=to_gps(start), end=to_gps(stop))
    for name in data:
        plt = data[name].plot(figsize=(100, 10))
        # modify the figure as a whole.
        plt.add_segments_bar(dq, label='')
        plt.gca().set_xscale('days')
        plt.suptitle(name)
        plt.save(f'{name}.png')


def plot_range():
    from channeling import channeling_reader
    from gwpy.segments import DataQualityFlag
    from gwpy.time import to_gps
    downloader = channeling_reader(['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], start, search_dirs=['../darm'])
    starte = start + timedelta(days=13)
    stop = starte + timedelta(days=2)
    dqflag = 'L1:DMT-ANALYSIS_READY:1'
    dq = DataQualityFlag.query(dqflag, to_gps(starte), to_gps(stop))
    data = downloader(['L1:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean'], start=to_gps(starte), end=to_gps(stop))
    for name in data:
        plt = data[name].plot()  # figsize=(1, 10))
        # modify the figure as a whole.
        plt.add_segments_bar(dq, label='')
        plt.gca().set_xscale('hours')
        plt.suptitle(name)
        plt.save(f'{name}.png')

def plot_spectra(clusters, channel, unit='cts', xlog=True, legend=None, xlim=None, **kwargs):
    from glob import glob
    from gwpy.frequencyseries import FrequencySeries
    from gwpy.plot import Plot
    title = channel
    psds = {}
    for cluster in clusters:
        for filename in glob('*.hdf5'):
            try:
                psds[cluster] = FrequencySeries.read(filename, f'{cluster}-{channel}')
                print(f'found in {filename}')
                break
            except KeyError:
                continue
        else:
            raise KeyError(f'Could not find Nº{cluster}')

    if legend is None:
        legend = clusters

    # plot the group in one figure.
    plt = Plot(*(psds[cluster] for cluster in psds), separate=False, sharex=True, zorder=1, **kwargs)
    if xlim is not None:
        plt.gca().set_xlim(xlim)
    plt.gca().set_ylim((1e-48, 1e-37))
    # modify the figure as a whole.
    # plt.add_segments_bar(dq, label='')
    # plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
    if xlog:
        plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.gca().set_ylabel(f'Power Spectral Density [{unit}^2/Hz]')
    plt.suptitle(title)
    plt.legend(legend, prop={'size': 15})

    # save to png.
    plt.save(f'{title}.png')

if __name__ == '__main__':
    from sys import argv

    exec(argv[-1])
