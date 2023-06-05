import scipy
import numpy as np




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