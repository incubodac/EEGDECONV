import scipy
import numpy as np

from art import *



def create_design_matrix(y_eeg,n_predictors,n_samples_window,timelimits,sr):
    #building design matrix from scratch
    #n_predictors = 3 
    #n_samples_window = 307
    #timelimits = [-.2,.4] #307  samples per predictor * 4
    #sr = 512

    expanded_params = (1+n_predictors)*n_samples_window
    signal_longitud_in_samples = len(y_eeg)

    X = np.zeros([signal_longitud_in_samples, expanded_params])

    x         = np.linspace(timelimits[0],timelimits[1],n_samples_window)

    zero_idx=closest_indices(x,0)
    evt_lat = df_conditions.iloc[:,-1].values

    for beta in range(n_predictors+1):
        
        j_idx = np.arange(beta*n_samples_window,beta*n_samples_window+n_samples_window)   
        for j in j_idx:   
            for i in range(len(evt_lat)):
                X[evt_lat[i]+j-beta*n_samples_window-zero_idx,j] = df_conditions.iloc[:,beta].values[i]
    return X

import mne
import pandas as pd
import numpy as np
import pathlib
import os
import logging
import datetime


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
        self.results_path = paths().results_path()
        self.log_path = paths().log_path()

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

    def initialize_logging(self):
    # Initialize logging configuration
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_level = logging.INFO
        log_filename = 'analysis_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.log'
        
        
        logger = logging.getLogger()
        logger.setLevel(log_level)
        # Create a new log file for each run
        handler = logging.FileHandler( os.path.join(self.log_path, log_filename), mode='w')
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
        #logging.basicConfig(filename=os.path.join(self.log_path, log_filename), level=log_level, format=log_format)
        logger.info("-----------------------------------------------------")
        logger.info("------------LOG INITIALIZATION-----------------------")
        logger.info("-----------------------------------------------------")


import os

class paths:
    """
    Paths to participants data.

    Attributes
    -------
    name: str
        Runing computers name result of os.popen('whoami').read().
    """

    def __init__(self):
        self.name = os.popen('whoami').read()

        if self.name == 'dac\n':
            if os.path.exists('/media/dac/SSD-curie'):
                self.main_path = '/media/dac/SSD-curie/Hybrid/'
            elif os.path.exists('/Volumes/DAC1T/'):
                self.main_path = '/Volumes/DAC1T/Hybrid/'
            else:
                print('HDD not connected\n')



        elif self.name == '':
            self.main_path = ''


    def eeg_analysis_path(self):
        """       
        Paths to participants EEG data to analyze.

        Returns
        -------
        eeg_path: str
            Path in str format to the .set and .fdt EEG data.
        """

        eeg_path = self.main_path + 'Hybrid_preana_out/8.data_analysis/'

        return eeg_path

    def eeg_raw_path(self):
        """       
        Paths to participants raw EEG data.

        Returns
        -------
        eeg_path: str
            Path in str format to the  EEG bdf data and the 
            bh data tables.
        """

        raw_path = self.main_path + 'HybridSearch/'

        return raw_path

    def evts_path(self):
        """
        Paths to participants event eeglab struct data
        
        (MATLAB).

        Returns
        -------
        et_path: str
            Path in str format to the ET folder containing the Eye-Tracker data.
        """

        evts_path = self.main_path + 'Hybrid_preana_out/9.event_structs/'

        return evts_path





    def results_path(self):
        """
        Paths to the results folder.

        Returns
        -------
        results_path: str
            Path in str format to the folder to store the results.
        """

        results_path = self.main_path + 'Hybrid_ana_out/'

        # Create directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)

        return results_path


    def plots_path(self):
        """
        Paths to the plots folder.

        Returns
        -------
        plots_path: str
            Path in str format to the folder to store the results.
        """

        plots_path = self.main_path + 'Hybrid_ana_out/plots/'

        # Create directory if it doesn't exist
        os.makedirs(plots_path, exist_ok=True)

        return plots_path


    def item_pos_path(self):
        """
        Paths to the search items positions file.

        Returns
        -------
        item_pos_path: str
            Path in str format to the items positions file.
        """

        item_pos_path = self.main_path + 'HybridSearch/posiciones_estimulos_completa_2022.csv'

        return item_pos_path


    def experiment_path(self):
        """
        Paths to the Psychopy experiment directory.

        Returns
        -------
        exp_path: str
            Path in str format to the items positions file.
        """

        exp_path = self.main_path + 'HybridSearch/Psychopy experimento/'

        return exp_path
    
    def log_path(self):
        """
        Paths to the logs folder

        Returns
        -------
        log_path: str
            Path in str format to the folder to store the log files.
        """

        log_path = self.main_path + 'Hybrid_ana_out/logs/'

        # Create directory if it doesn't exist
        os.makedirs(log_path, exist_ok=True)
        
        return log_path
    
    def full_metadata_path(self):
        """
        Paths to the logs folder

        Returns
        -------
        log_path: str
            Path in str format to the folder to store the log files.
        """

        full_metadata_path = self.main_path + 'Hybrid_ana_out/metadata/'

        # Create directory if it doesn't exist
        os.makedirs(full_metadata_path, exist_ok=True)
        
        return full_metadata_path





class subject:
    """
    Class containing methods to get eeg,et and bh 
    data for a subject.

    Attributes
    -------
    _path: str
        Path to the EEG data.
    subjects_ids: list
        List of subject's id.

    subjects_groups: list
        List of subject's group
    """
    def __init__(self,exp_info,subject_code=None):
        if subject_code == None:
            subject_id = exp_info.subjects_ids[0]
        elif type(subject_code) == int:
            subject_id = exp_info.subjects_ids[subject_code]
        elif type(subject_code) == str and (subject_code in exp_info.subjects_ids):
            subject_id = subject_code
        else:
            print('Subject not found.')
            
        self.subject_id   = subject_id
        logger = logging.getLogger()
        logger.info("Class containing subject %s information was loaded",self.subject_id)
        logger.info("-----------------------------------------------------")

    
    def load_bh_csv(self):
        bh_csv_path = paths().eeg_raw_path() + self.subject_id + '/'
        files = os.listdir(bh_csv_path)
        # Filter the list to include only CSV files
        csv_files = [f for f in files if f.endswith('.csv') and not f.startswith('.')]
        if len(csv_files) == 1:
            # Read the CSV file into a DataFrame
            tprint(f'''\nLoading behavioural data.....
            from   {csv_files}\n''',font="fancy67")
            csv_path = os.path.join(bh_csv_path, csv_files[0])
            df = pd.read_csv(csv_path)
            logger = logging.getLogger()
            logger.info("Behaviour csv from subject %s was loaded",self.subject_id)
            logger.info("-----------------------------------------------------")

            return df
        else:
            print('Error: There is not exactly one CSV file in the folder.\n')
            print(csv_files)
            

    def load_analysis_eeg(self):
        tprint(f'''\nLoading EEG data.....
        subject   {self.subject_id}\n''',font="fancy67")
        # get subject path
        set_path =  paths().eeg_analysis_path()
        set_file =  os.path.join(set_path,f'{self.subject_id}_analysis.set')

        # Load sesions
        try:
            raw     = mne.io.read_raw_eeglab( set_file, preload=True)
            return raw
        # Missing data
        except FileNotFoundError:
            print('No .ds files found in directory: {}'.format(set_path))

    def load_event_struct(self):
        tprint('''\nLoading events data.....
        from\n''',font="fancy67")        # get subject path
        evts_path =  paths().evts_path()
        evts_file =  os.path.join(evts_path,f'{self.subject_id}_events.csv')
        tprint(evts_file +'\n',font="fancy67")
        # Load sesions
        try:
            df = pd.read_csv(evts_file)
            logger = logging.getLogger()
            logger.info("Events csv from subject %s was loaded",self.subject_id)
            logger.info("-----------------------------------------------------")
            return df
        # Missing data
        except FileNotFoundError:
            print('No event files found in directory: {}'.format(evts_path))


    def get_et_data(self,raw,sample_limits,plot=0):
        '''
        sample_limits = [start_sample, stop_sample] list
        plot 1/0
        '''
        et_chans = setup.exp_info().et_channel_names
        et       = raw[et_chans,sample_limits[0]:sample_limits[1]]
        x = et[0].T[:,0]
        y = et[0].T[:,1]

        if plot:
            plt.plot(x,y)
            plt.show()
        return x , y 
    
    def load_electrode_positions(self,raw):
        montage = mne.channels.make_standard_montage('biosemi128')
        raw.set_montage(montage, on_missing='ignore')
        return raw
    
    def load_metadata(self):
        tprint('''\nLoading events data.....
        from\n''',font="fancy67")        # get subject path
        sub_id = self.subject_id
        metadata_path =  paths().full_metadata_path()
        evts_file =  os.path.join(metadata_path,f'{sub_id}_full_metadata.csv')
        tprint(evts_file +'\n',font="fancy67")
        # Load sesions
        try:
            df = pd.read_csv(evts_file)
            logger = logging.getLogger()
            logger.info("Events csv from subject %s was loaded",self.subject_id)
            logger.info("-----------------------------------------------------")
            return df
        # Missing data
        except FileNotFoundError:
            print('No event files found in directory: {}'.format(evts_path))
