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
            if not os.path.exists('/Volumes/DAC1T/'):
                print('HDD not connected\n')
            else:
                '''print('HDD found\n')'''
            self.main_path = '/Volumes/DAC1T/Hybrid/'


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


