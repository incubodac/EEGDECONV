from paths import paths
import mne
import pandas as pd
import numpy as np
import pathlib
import os


class exp_info:
    """
    Class containing the experiment information.

    Attributes
    -------
    _path: str
        Path to the EEG data.
    subjects_ids: list
        List of subject's id.

    subjects_groups: list
        List of subject's group
    """

    def __init__(self):
    # Define set and fdt path and et data path
        self.eeg_path = paths().eeg_analysis_path()
        self.raw_path = paths().eeg_raw_path()

        # Select subject
        self.subjects_ids = ['S101','S102','S103','S104','S105','S106','S107','S108','S109','S110','S111',
                             'S112','S113','S114','S115','S116','S117','S118','S119']


        # Subjects interpolated channels
        self.subjects_int_channels = {'S101': [], 'S102': [], 'S102': [], 'S103': [], 'S104': [], 'S105': [],
                                      'S106': [], 'S107': [], 'S108': [], 'S109': [], 'S110': [], 'S111': [],
                                      'S112': [], 'S113': [], 'S114': [], 'S115': [], 'S116': [], 'S117': [],
                                      'S118': [], 'S119': []}

        # Subject rejected ICA components based on plochl criteria and visual inspection
        self.subjects_rej_ic = {'S101': [], 'S102': [], 'S102': [], 'S103': [], 'S104': [], 'S105': [],
                                'S106': [], 'S107': [], 'S108': [], 'S109': [], 'S110': [], 'S111': [],
                                'S112': [], 'S113': [], 'S114': [], 'S115': [], 'S116': [], 'S117': [],
                                'S118': [], 'S119': []}

        # Missing et data
        self.no_trig_subjects = []


        # Get et channels by name [Gaze x, Gaze y, Pupils]
        self.et_channel_names = ['R-GAZE-X', 'R-GAZE-Y', 'R-AREA']

        # Trigger channel name
        self.trig_ch = ''
 

        self.line_noise_freqs = (50)
        
        self.screen_size = (1920,1080)

        
        


class config:
    """
    Class containing the run configuration.

    Attributes
    -------

    subjects_ids: list
        List of subject's id.
    subjects_bad_channels: list
        List of subject's bad channels.
    subjects_groups: list
        List of subject's group
    missing_bh_subjects: list
        List of subject's ids missing behavioural data.
    trials_loop_subjects: list
        List of subject;s ids for subjects that took the firts version of the experiment.
    """

    def __init__(self):
        self.preprocessing = self.preprocessing()
        self.general = self.general()

    class preprocessing:
        def __init__(self):

            # Distance to the screen during the experiment
            self.screen_distance = {'S101': 58, 'S102': 58, 'S103': 58, 'S104': 58, 'S105': 58,
                                    'S106': 58, 'S107': 58, 'S108': 58, 'S109': 58, 'S110': 58,
                                    'S111': 58, 'S112': 58, 'S113': 58, 'S114': 58, 'S115': 58,
                                    'S116': 58, 'S117': 58, 'S118': 58, 'S119': 58}

            
    class general:
        def __init__(self):
            #Reject windows parameter based on EEG peak to peak amplitude
            self.reject_amp =  {'S101': 250e-6, 'S102': 250e-6, 'S103': 250e-6, 'S104': 250e-6, 'S105': 250e-6,
                                'S106': 250e-6, 'S107': 250e-6, 'S108': 250e-6, 'S109': 250e-6, 'S110': 250e-6,
                                'S111': 250e-6, 'S112': 250e-6, 'S113': 250e-6, 'S114': 250e-6, 'S115': 250e-6,
                                'S116': 250e-6, 'S117': 250e-6, 'S118': 250e-6, 'S119': 250e-6}

class eeg_subject:
    """
    Class containing subjects data.

    Parameters
    ----------
    subject: {'int', 'str'}, default=None
        Subject id (str) or number (int). If None, takes the first subject.

    Attributes
    ----------
    bad_channels: list
        List of bad channels.
    beh_path: str
        Path to the behavioural data.
    subject_id: str
        Subject id.
    """

    def __init__(self, exp_info, config, subject_code=None):

        # Select 1st subject by default
        if subject_code == None:
            self.subject_id = exp_info.subjects_ids[0]
        # Select subject by index
        elif type(subject_code) == int:
            self.subject_id = exp_info.subjects_ids[subject_code]
        # Select subject by id
        elif type(subject_code) == str and (subject_code in exp_info.subjects_ids):
            self.subject_id = subject_code
        else:
            print('Subject not found')

        # Subject's data paths
        self.eeg_path = pathlib.Path(os.path.join(exp_info.eeg_path, self.subject_id))
        self.bh_path  = pathlib.Path(os.path.join(exp_info.bh_path, self.subject_id))

        # Define subject group and bad channels by matching id index
        self.bad_channels = exp_info.subjects_bad_channels[self.subject_id]
        self.group = exp_info.subjects_groups[self.subject_id]

        # Define mapping between button value and color by group
        if self.group == 'Balanced':
            self.map = {'blue': '1', 'red': '4'}
        elif self.group == 'Counterbalanced':
            self.map = {'blue': '4', 'red': '1'}

        # Get run configuration for subject
        self.config = self.subject_config(config=config, subject_id=self.subject_id)


    # Subject's parameters and configuration
    class subject_config:

        def __init__(self, config, subject_id):
            self.preproc = self.preproc(config=config, subject_id=subject_id)
            self.general = self.general(config=config, subject_id=subject_id)

        # Configuration for preprocessing run
        class preproc:
            def __init__(self, config, subject_id):

                # Get config.preprocessing attirbutes and get data for corresponding subject
                preproc_attributes = config.preprocessing.__dict__.keys()

                # Iterate over attributes and get data for conrresponding subject
                for preproc_att in preproc_attributes:
                    att = getattr(config.preprocessing, preproc_att)
                    if type(att) == dict:
                        try:
                            # If subject_id in dictionary keys, get attribute, else pass
                            att_value = att[subject_id]
                            setattr(self, preproc_att, att_value)
                        except:
                            pass
                    else:
                        # If attribute is general for all subjects, get attribute
                        att_value = att
                        setattr(self, preproc_att, att_value)

        # Configuration for further analysis
        class general:
            def __init__(self, config, subject_id):

                # Get config.preprocessing attirbutes and get data for corresponding subject
                general_attributes = config.general.__dict__.keys()

                # Iterate over attributes and get data for conrresponding subject
                for general_att in general_attributes:
                    att = getattr(config.general, general_att)
                    if type(att) == dict:
                        try:
                            # If subject_id in dictionary keys, get attribute, else pass
                            att_value = att[subject_id]
                            setattr(self, general_att, att_value)
                        except:
                            pass
                    else:
                        # If attribute is general for all subjects, get attribute
                        att_value = att
                        setattr(self, general_att, att_value)


    # MEG data
    def load_raw_meg_data(self):
        """
        MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading MEG data')
        # get subject path
        subj_path = self.ctf_path
        ds_files = list(subj_path.glob('*{}*.ds'.format(self.subject_id)))
        ds_files.sort()

        # Load sesions
        # If more than 1 session concatenate all data to one raw data
        if len(ds_files) > 1:
            raws_list = []
            for i in range(len(ds_files)):
                raw = mne.io.read_raw_ctf(ds_files[i], system_clock='ignore')
                raws_list.append(raw)
            # MEG data structure
            raw = mne.io.concatenate_raws(raws_list, on_mismatch='ignore')
            return raw
        # If only one session return that session as whole raw data
        elif len(ds_files) == 1:
            raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')
            return raw
        # Missing data
        else:
            raise ValueError('No .ds files found in subject directory: {}'.format(subj_path))


    # ET data
    def load_raw_et_data(self):
        """
        load ET parseeyelink matlab struct
        """
        print('\nLoading Preprocessed et data')
        
        # get subject path
        et_path = paths().et_path()
        file_path = pathlib.Path(os.path.join(preproc_path, self.subject_id, f'Subject_{self.subject_id}_meg.fif'))

        # Load data
        fif = mne.io.read_raw_fif(file_path, preload=preload)

        return fif
        



    # Behavioural data
    def load_raw_bh_data(self):
        """
        Behavioural data for parent subject as pandas DataFrames.
        """
        # Get subject path
        subj_path = self.bh_path
        bh_file = list(subj_path.glob('*.csv'.format(self.subject_id)))[0]

        # Load DataFrame
        df = pd.read_csv(bh_file)

        return df


    # MEG data
    def load_preproc_meg(self, preload=False):
        """
        Preprocessed MEG data for parent subject as raw instance of MNE.
        """

        print('\nLoading Preprocessed MEG data')
        # get subject path
        preproc_path = paths().preproc_path()
        file_path = pathlib.Path(os.path.join(preproc_path, self.subject_id, f'Subject_{self.subject_id}_meg.fif'))

        # Load data
        fif = mne.io.read_raw_fif(file_path, preload=preload)

        return fif





class all_subjects:

    def __init__(self, all_fixations, all_saccades, all_bh_data, all_rt, all_corr_ans):
        self.subject_id = 'All_Subjects'
        self.fixations = all_fixations
        self.saccades = all_saccades
        self.trial = np.arange(1, 211)
        self.bh_data = all_bh_data
        self.rt = all_rt
        self.corr_ans = all_corr_ans
        
        
if __name__=='__main__':
    exp_info().et_channel_names()