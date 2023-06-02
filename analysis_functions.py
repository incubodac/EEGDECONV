import mne
import numpy as np
import functions_general
import matplotlib.pyplot as plt
from paths import paths
import os
import time
import save
import load
import setup
import paths
#DAC code

    


#Joaco code
def define_events(subject, meg_data, epoch_id, mss=None, trials=None, screen=None, dur=None, tgt=None, dir=None, evt_from_df=False):

    print('Defining events')

    metadata_sup = None

    if evt_from_df:
        if 'fix' in epoch_id:
            metadata = subject.fixations
        elif 'sac' in epoch_id:
            metadata = subject.saccades

        # Get events from fix/sac Dataframe
        if screen:
            metadata = metadata.loc[(metadata['screen'] == screen)]
        if mss:
            metadata = metadata.loc[(metadata['mss'] == mss)]
        if dur:
            metadata = metadata.loc[(metadata['duration'] >= dur)]
        if 'fix' in epoch_id:
            if tgt == 1:
                metadata = metadata.loc[(metadata['fix_target'] == tgt)]
            elif tgt == 0:
                metadata = metadata.loc[(metadata['fix_target'] == tgt)]
        if 'sac' in epoch_id:
            if dir:
                metadata = metadata.loc[(metadata['dir'] == dir)]

        metadata.reset_index(drop=True, inplace=True)

        events_samples, event_times = functions_general.find_nearest(meg_data.times, metadata['onset'])

        events = np.zeros((len(events_samples), 3)).astype(int)
        events[:, 0] = events_samples
        events[:, 2] = metadata.index

        events_id = dict(zip(metadata.id, metadata.index))

    else:
        # Get events from annotations
        all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)

        # Select epochs
        epoch_keys = [key for key in all_event_id.keys() if epoch_id in key]
        if 'sac' not in epoch_id:
            epoch_keys = [key for key in epoch_keys if 'sac' not in key]
        if 'fix' not in epoch_id:
            epoch_keys = [key for key in epoch_keys if 'fix' not in key]
        # if screen:
        #     epoch_keys = [epoch_key for epoch_key in epoch_keys if f'{screen}' in epoch_key]
        if trials != None:
            try:
                epoch_keys = [epoch_key for epoch_key in epoch_keys if
                              (epoch_key.split('_t')[-1].split('_')[0] in trials and 'end' not in epoch_key)]
            except:
                print('Trial selection skipped. Epoch_id does not contain trial number.')
        # if mss:
        #     trials_mss = subject.bh_data.loc[subject.bh_data['Nstim'] == mss].index + 1  # add 1 due to python 0th indexing
        #     epoch_keys = [epoch_key for epoch_key in epoch_keys if int(epoch_key.split('t')[-1]) in trials_mss]

        # Get events and ids matchig selection
        metadata, events, events_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id,
                                                               row_events=epoch_keys, tmin=0, tmax=0,
                                                               sfreq=meg_data.info['sfreq'])

        if 'fix' in epoch_id:
            metadata_sup = subject.fixations
        elif 'sac' in epoch_id:
            metadata_sup = subject.saccades

    return metadata, events, events_id, metadata_sup


def epoch_data(subject, mss, corr_ans, tgt_pres, epoch_id, meg_data, tmin, tmax, baseline=(None, 0), reject=None,
               save_data=False, epochs_save_path=None, epochs_data_fname=None):
    '''
    :param subject:
    :param mss:
    :param corr_ans:
    :param tgt_pres:
    :param epoch_id:
    :param meg_data:
    :param tmin:
    :param tmax:
    :param baseline: tuple
    Baseline start and end times.
    :param reject: float|str|bool
    Peak to peak amplituyde reject parameter. Use 'subject' for subjects default calculated for short fixation epochs.
     Use False for no rejection. Default to 4e-12 for magnetometers.
    :param save_data:
    :param epochs_save_path:
    :param epochs_data_fname:
    :return:
    '''

    # Sanity check to save data
    if save_data and (not epochs_save_path or not epochs_data_fname):
        raise ValueError('Please provide path and filename to save data. Else, set save_data to false.')

    # Trials
    cond_trials, bh_data_sub = functions_general.get_condition_trials(subject=subject, mss=mss,
                                                                      corr_ans=corr_ans, tgt_pres=tgt_pres)
    # Define events
    metadata, events, events_id, metadata_sup = define_events(subject=subject, epoch_id=epoch_id,
                                                                                 trials=cond_trials,
                                                                                 meg_data=meg_data)
    # Reject based on channel amplitude
    if reject == None:
        # Not setting reject parameter will set to default subject value
        reject = dict(mag=4e-12)
    elif reject == False:
        # Setting reject parameter to False uses No rejection (None in mne will not reject)
        reject = None
    elif reject == 'subject':
        reject = dict(mag=subject.config.general.reject_amp)

    # Epoch data
    epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                        event_repeated='drop', metadata=metadata, preload=True, baseline=baseline)
    # Drop bad epochs
    epochs.drop_bad()

    if metadata_sup is not None:
        metadata_sup = metadata_sup.loc[(metadata_sup['id'].isin(epochs.metadata['event_name']))].reset_index(drop=True)
        epochs.metadata = metadata_sup

    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        os.makedirs(epochs_save_path, exist_ok=True)
        epochs.save(epochs_save_path + epochs_data_fname, overwrite=True)

    return epochs, events


def time_frequency(epochs, l_freq, h_freq, freqs_type, n_cycles_div=4., return_itc=True, save_data=False, trf_save_path=None,
                   power_data_fname=None, itc_data_fname=None):

    # Sanity check to save data
    if save_data and (not trf_save_path or not power_data_fname or not itc_data_fname):
        raise ValueError('Please provide path and filename to save data. Else, set save_data to false.')

    # Compute power over frequencies
    print('Computing power and ITC')
    if freqs_type == 'log':
        freqs = np.logspace(*np.log10([l_freq, h_freq]), num=40)
    elif freqs_type == 'lin':
        freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)  # 1 Hz bands
    n_cycles = freqs / n_cycles_div  # different number of cycle per frequency
    power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                               return_itc=return_itc, decim=3, n_jobs=None, verbose=True)

    if save_data:
        # Save trf data
        os.makedirs(trf_save_path, exist_ok=True)
        power.save(trf_save_path + power_data_fname, overwrite=True)
        itc.save(trf_save_path + itc_data_fname, overwrite=True)

    return power, itc


def get_plot_tf(tfr, plot_xlim=(None, None), plot_max=True, plot_min=True):
    if plot_xlim:
        tfr_crop = tfr.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1])
    else:
        tfr_crop = tfr.copy()

    timefreqs = []

    if plot_max:
        max_ravel = tfr_crop.data.mean(0).argmax()
        freq_idx = int(max_ravel / len(tfr_crop.times))
        time_percent = max_ravel / len(tfr_crop.times) - freq_idx
        time_idx = round(time_percent * len(tfr_crop.times))
        max_timefreq = (tfr_crop.times[time_idx], tfr_crop.freqs[freq_idx])
        timefreqs.append(max_timefreq)

    if plot_min:
        min_ravel = tfr_crop.data.mean(0).argmin()
        freq_idx = int(min_ravel / len(tfr_crop.times))
        time_percent = min_ravel / len(tfr_crop.times) - freq_idx
        time_idx = round(time_percent * len(tfr_crop.times))
        min_timefreq = (tfr_crop.times[time_idx], tfr_crop.freqs[freq_idx])
        timefreqs.append(min_timefreq)

    timefreqs.sort()

    return timefreqs


def create_mne_events