

def start_stop_samples_trigg(evts,trigg):
    trigg_samples = evts[evts['type']==trigg]['latency'].to_numpy().astype(int)
    dur_samples   = evts[evts['type']==trigg]['duration'].to_numpy().astype(int)
    
    return trigg_samples, trigg_samples + dur_samples


def start_samples_trigg(evts,trigg):
    
    if type(trigg)==str:
        trigg_samples = evts[evts['type']==trigg]['latency'].to_numpy().astype(int)
    elif type(trigg)==list:
        ev   = evts.loc[evts['type'].isin(trigg)]
        trigg_samples = ev['latency'].to_numpy().astype(int)
    
    return trigg_samples

def expand_evts_struct(evts):
    
    
    return evts

def closest_tuple(tuples, threshold, point):
    import numpy as np
    from scipy.spatial.distance import cdist
    distances = cdist(np.array(tuples), np.array([point]))
    within_threshold = distances < threshold
    if np.any(within_threshold):
        closest_idx = np.argmin(distances[within_threshold])
        closest_tuple_idx = np.where(within_threshold)[0][closest_idx]
        return True, closest_tuple_idx
    else:
        return False, None



def add_trial_info_to_events(evts,bh_data):
    import numpy as np
    def key2bool(val):
        if val == 'right':
            return True
        elif val == 'left':
            return False
        else:
            return val

    #modify evts dataframe 
    column = ['phase','istarget','isdistractor','']
    phases = {'cross1','mem','cross2','vs','bad_ET'}
    emvs   = {'fixation','saccade'}
    tr=0
    evts['trial']          = np.nan
    evts['phase']          = np.nan
    evts['mss']            = np.nan
    evts['ontarget']       = np.nan
    evts['ondistractor']   = np.nan
    evts['present']        = np.nan
    evts['correct']        = np.nan
    msss     = list(bh_data.loc[::6,'Nstim'])
    presents = list((bh_data.loc[::6,'st5_cat']=='T' ) & ~(bh_data.loc[::6,'st5']=='memstim/dog1962.png'))
    pressed = bh_data.loc[5::6,'key_resp.keys']
    corrects = list(presents[:-1]== pressed.map(key2bool))


    #define start and stop latencies for phase events then loop over events and label them
    cross1_start_samp, cross1_stop_samp = start_stop_samples_trigg(evts,'cross1')
    mem_start_samp, mem_stop_samp       = start_stop_samples_trigg(evts,'mem')
    cross2_start_samp, cross2_stop_samp = start_stop_samples_trigg(evts,'cross2')
    vs_start_samp, vs_stop_samp         = start_stop_samples_trigg(evts,'vs')

    for index, row  in evts.iterrows():
        if evts.at[index,'type']=='cross1':
            tr+=1
        elif evts.at[index,'type'] in emvs:
            evts.at[index,'trial'] = tr
            evts.at[index,'mss']   = msss[tr-1]
            evts.at[index,'present']   = presents[tr-1]
            evts.at[index,'correct']   = corrects[tr-1]
            
            if cross1_start_samp[tr-1] < evts.at[index,'latency'] < cross1_stop_samp[tr-1]:
                evts.at[index,'phase'] = 'cross1'
            elif mem_start_samp[tr-1] < evts.at[index,'latency'] < mem_stop_samp[tr-1]:
                evts.at[index,'phase'] = 'mem'
            elif cross2_start_samp[tr-1] < evts.at[index,'latency'] < cross2_stop_samp[tr-1]:
                evts.at[index,'phase'] = 'cross2'
            elif vs_start_samp[tr-1] < evts.at[index,'latency'] < vs_stop_samp[tr-1]:
                evts.at[index,'phase'] = 'vs'
                                 
    return evts
   

def plot_fix_durs_mem_vs(evts):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    main_phases = [ 'mem', 'vs']
    for phase in main_phases:
        ev = evts[(evts['type'] == 'fixation') & (evts['phase'] == phase)]
        n_fixations = len(ev)
        ax.hist(ev['duration'], bins=80, alpha=0.5, density=True, label=f'{phase} (N={n_fixations})')

    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Density')
    ax.set_title('Normalized Fixation Duration Distribution')
    ax.legend()
    plt.show()

def plot_fix_durs_all_phases(evts):
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    axs = axs.flatten()
    phas = ['cross1','mem','cross2','vs']
    for i in range(4):
        ev     = evts[(evts['type']=='fixation') & (evts['phase']==phas[i])]
        n_fixations = len(ev)
        axs[i].hist(ev['duration'], bins=80)
        axs[i].set_xlabel('Duration (ms)')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(phas[i])
        axs[i].legend([f'N={n_fixations}'])

    plt.tight_layout()
    plt.show()  
     


if __name__ == '__main__':
    import setup
    import load
    info = setup.exp_info()
    suj  = load.subject(info,0)
    evts = suj.load_event_struct()
    bh_data = suj.load_bh_csv()
    #lats=start_samples_trigg(info,suj,['mem','vs'])
    #print(len(lats))
    # tuples = [(2, 0), (5, 5), (5, 7)]
    # threshold = 2.5
    # point = (5, 6)
    # closest= closest_tuple(tuples, threshold, point)
    # print(closest)  # Output: (1, 2)
    evts = add_trial_info_to_events(evts,bh_data)
    plot_fix_durs_mem_vs(evts)

