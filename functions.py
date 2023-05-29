import logging
import load
import os



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



def add_trial_info_to_events(evts,bh_data,thr):
    from paths import paths
    import setup
    import matplotlib.image as mpimg
    import pandas as pd


    threshold = thr
    image_names = bh_data['searchimage'].drop_duplicates().str.split('cmp_', expand=True)[1].str.split('.jpg', expand=True)[0].to_list()
    path =    paths()
    exp_path = path.experiment_path()
    targets  = bh_data.loc[::6,['st5']] #first column has image name and second T/A(absent)
    target_files = targets['st5'].str.lstrip('memstim').str.lstrip('/')[:-1] #target filenames
    exp_path = path.experiment_path()
    info = setup.exp_info()
    screensize = info.screen_size#[ 1920,1080 ]
    item_pos = path.item_pos_path()
    df = pd.read_csv(item_pos)

        
    
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
    evts['stm']            = np.nan

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
            image_name   = image_names[tr-1]
            img = mpimg.imread(exp_path + 'cmp_' + image_name + '.jpg')
            xim = (screensize[0]-img.shape[1])/2
            yim = (screensize[1]-img.shape[0])/2
            trial_stims  = df[df['folder']==image_name]
    
            records = trial_stims.to_records(index=False)
            item_pos = [(record[6]+record[5]/2, record[7]+record[4]/2) for record in records]
            if presents[tr-1]:
                target_pos = trial_stims[trial_stims['stm']==target_files.iloc[tr-1]][['height','width','pos_x','pos_y']].to_records(index=False)
                target_pos = target_pos[0]
                target_pos = target_pos[2]+target_pos[1]/2,target_pos[3]+target_pos[0]/2
            else:
                target_pos = None    
            try:
                if presents[tr-1]:
                    if not target_pos:
                        raise ValueError("ima_pos is empty, but flag is True")
                else:
                    if target_pos:
                        raise ValueError("ima_pos is not empty, but flag is False")
            except ValueError as e:
                print(f"Sanity check failed: {str(e)}")
                
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
                point = (evts.at[index,'fix_avgpos_x']-xim,evts.at[index,'fix_avgpos_y']-yim)
            
                flag, closest_id = closest_tuple(item_pos, threshold, point)
                if not flag:
                    continue
                elif item_pos[closest_id]==target_pos:
                    evts.at[index,'ontarget']      = True
                    evts.at[index,'ondistractor']  = False
                    evts.at[index,'stm']           = trial_stims['stm'].iloc[closest_id]
                else:
                    evts.at[index,'ontarget']      = False
                    evts.at[index,'ondistractor']  = True
                    evts.at[index,'stm']           = trial_stims['stm'].iloc[closest_id]
    
    cross1_counts = len(evts[(evts['type']=='fixation') & (evts['phase']=='cross1')])
    mem_counts    = len(evts[(evts['type']=='fixation') & (evts['phase']=='mem')])
    cross2_counts = len(evts[(evts['type']=='fixation') & (evts['phase']=='cross2')])
    vs_counts     = len(evts[(evts['type']=='fixation') & (evts['phase']=='vs')])
    answer_acc = 100*sum(corrects)/len(corrects)
    
    print(f'percentage of correct answers : {answer_acc:.1f}')
    print(f'fixations in cross1 phase : {cross1_counts}\n')
    print(f'fixations in mem phase    : {mem_counts}\n')
    print(f'fixations in cross2 phase : {cross2_counts}\n')
    print(f'fixations in vs phase     : {vs_counts}\n')
    print(evts['type'].value_counts())
    total_captured_fixs = sum((evts['ondistractor']) | (evts['ontarget']))
    total_item_fixed = 100*total_captured_fixs/vs_counts
    on_targets  = sum((evts['ontarget']==True))
    on_distractors = sum((evts['ondistractor']==True))
    

      
    print(f'total fixations on items    : {total_captured_fixs}')
    print(f'fixations on targets  : {on_targets}')
    print(f'fixations on distractors  : {on_distractors}') 
    print(f'percentage of capture fixations in vs {total_item_fixed:.1f}%')

    logger = logging.getLogger()
    logger.info("Percentage of correct answers: %.1f %%", answer_acc)
    logger.info("Cross1: %d", cross1_counts)
    logger.info("Mem: %d", mem_counts)
    logger.info("Cross2: %d", cross2_counts)
    logger.info("VS: %d", vs_counts)
    logger.info("Total fixations on items (vs): %d", total_captured_fixs)
    logger.info("Total fixations on targets: %d", on_targets)
    logger.info("Total fixations on distractors: %d", on_distractors)
    logger.info("Percentage of capture fixations (vs): %.1f %%", total_item_fixed)

                        
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
     
def plot_trial(eeg,suj,tr):
    from paths import paths
    import setup
    import matplotlib.image as mpimg
    from matplotlib import pyplot as plt 
    import matplotlib.patches as patches
    import pandas as pd

    path =    paths()
    exp_path = path.experiment_path()
    item_pos = path.item_pos_path()
    #suj  = load.subject(info,0)
    info = setup.exp_info()
    screensize = info.screen_size
    bh_data     = suj.load_bh_csv()
    evts = suj.load_event_struct()
    image_names = bh_data['searchimage'].drop_duplicates()
    image_names = image_names.str.split('cmp_', expand=True)[1]
    image_names = image_names.str.split('.jpg', expand=True)[0]

    star_samp, stop_samp = start_stop_samples_trigg(evts,'vs')
    image_names = list(image_names)
    image_name  = image_names[tr-1]
    y,x = suj.get_et_data(eeg,[star_samp[tr-1],stop_samp[tr-1]])
    img = mpimg.imread(exp_path + 'cmp_' + image_name + '.jpg')

    xim = (screensize[0]-img.shape[1])/2
    yim = (screensize[1]-img.shape[0])/2

    fig, ax = plt.subplots()##############  SIZE
    ax.imshow(img)
    ax.plot(y*1e6-xim,x*1e6-yim,'black')

    #################check fix evt marks correspondence with scanpath#############
    fix_start_samps = start_samples_trigg(evts,'fixation')
    fixs_lats = [x for x in fix_start_samps if star_samp[tr-1]  <   x  <  stop_samp[tr-1]]



    for i in fixs_lats:
        xf = eeg[info.et_channel_names[0],i][0]
        yf = eeg[info.et_channel_names[1],i][0]
        ax.scatter(xf*1e6-xim,yf*1e6-yim,s=50,color='blue')
        ####add fixations positions from evts data###########
        #point = evts[evts['latency']==i][['fix_avgpos_x','fix_avgpos_y']]
        x_ev = evts[evts['latency']==i]['fix_avgpos_x']
        y_ev = evts[evts['latency']==i]['fix_avgpos_y']
        ax.scatter(x_ev-xim,y_ev-yim,s=40,color='g')


    #############################################################################
    df = pd.read_csv(item_pos)

    for index, row in df[df['folder']==image_name].iterrows():
    # Create a rectangle patch for the bounding box
        rect = patches.Rectangle((row['pos_x'], row['pos_y']), row['width'], row['height'], linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the rectangle patch to the plot
        ax.add_patch(rect)
    #########add centers#######
    #df is position csv image_name is searchimage
    trial_stims = df[df['folder']==image_name]
    records = trial_stims.to_records(index=False)
    # Extract the (x, y) values from each record as a tuple using a list comprehension
    centers_list = [(record[6]+record[5]/2, record[7]+record[4]/2) for record in records]
    #closest_tuple(centers_list, 40, (419,500))
    for i in range(len(centers_list)):
        plt.scatter(centers_list[i][0],centers_list[i][1],color='black')

def create_full_metadata(info,sub_id,metadata_path,capturing_thr,save_evts=False):
    suj  = load.subject(info, sub_id)
    eeg  = suj.load_analysis_eeg()
    #eeg  = suj.load_electrode_positions(eeg)
    evts = suj.load_event_struct()
    bh_data     = suj.load_bh_csv()
    evts = add_trial_info_to_events(evts,bh_data,capturing_thr)
    if save_evts:
        # Save epoched data     
        evts.to_csv(os.path.join(metadata_path,f'{sub_id}_full_metadata.csv'), index=False)
        logger = logging.getLogger()
        logger.info("saving full metadata for subject: %s ", sub_id)
    return evts

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
    #evts = add_trial_info_to_events(evts,bh_data)
    #plot_fix_durs_mem_vs(evts)

