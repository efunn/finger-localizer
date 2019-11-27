import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sea

SUBJ_NUMS = [1,2,3,4,6,7,8,10,11,12,13,14]
SUBJ_IDS = ['ft'+str(subj_num).zfill(3) for subj_num in SUBJ_NUMS]
BASELINE_TRS = 10
TR_TO_EXTRACT = 6
TRS_PER_TRIAL = 8
TRS_PER_RUN = 170
NUM_RUNS = 8
TRS_TO_EXTRACT = np.array([np.arange(BASELINE_TRS+TR_TO_EXTRACT-1,TRS_PER_RUN,NUM_RUNS)+TRS_PER_RUN*run
                           for run in range(NUM_RUNS)]).flatten()
PVAL_THRESH = 1e-4

# helper function for moving average
def moving_average(in_data, n_points=3):
    filt_b=np.ones(n_points)/float(n_points); filt_a=[1]
    out_data = lfilter(filt_b,filt_a,in_data,axis=0)
    return out_data

# calculate the DICE coefficient (overlap of 2 vectors)
def dice(vec1, vec2):
    intersection = np.logical_and(vec1, vec2)
    return 2.*intersection.sum()/(vec1.sum()+vec2.sum())

# sample for single subject analysis
for subj_id in SUBJ_IDS:
    fmri_data = moving_average(np.load('data/'+subj_id+'_localizer.npy'))[TRS_TO_EXTRACT,:]
    press_data = pd.read_csv('data/'+subj_id+'_localizer_press.csv')
    trial_data = press_data.groupby('trial',sort=False).first() 
    trial_labels = trial_data.target_finger
    finger_pattern_pvals = np.zeros((len(np.unique(trial_labels)),fmri_data.shape[1]))
    for finger in np.unique(trial_labels):
        _,finger_pattern_pvals[finger,:] = ttest_1samp(fmri_data[trial_labels==finger,:],0)
    print('--------------------------------')
    print('for subject: '+subj_id)
    print('index-middle overlap: '+str(dice(finger_pattern_pvals[0,:]<PVAL_THRESH,finger_pattern_pvals[1,:]<PVAL_THRESH)))
    print('middle-ring overlap: '+str(dice(finger_pattern_pvals[1,:]<PVAL_THRESH,finger_pattern_pvals[2,:]<PVAL_THRESH)))
    print('ring-little overlap: '+str(dice(finger_pattern_pvals[2,:]<PVAL_THRESH,finger_pattern_pvals[3,:]<PVAL_THRESH)))
    print('--------------------------------')

# sample hemodynamic response
# you'll need to convolve your experimental design with this
# e.g. using np.convolve(my_sample_signal,gen_hrf())
def gen_hrf(tr=2, n_trs=15, c=1./6, a1=6, a2=16):
    # a1, a2: timepoints of peaks
    # c: ratio between peak and trough
    t = tr*np.arange(n_trs) + tr*.5
    h = (np.exp(-t)*(t**(a1-1)/np.math.factorial(a1-1)
         - c*t**(a2-1)/np.math.factorial(a2-1)))
    return h/np.sum(h)


################
# RSA analysis #
################
import matplotlib.pyplot as plt
import seaborn as sea
import matplotlib.gridspec as gridspec

num_fingers = 4
trials_per_finger = 40
gs = gridspec.GridSpec(2,6)
axs = [plt.subplot(gs[0]),plt.subplot(gs[1]),plt.subplot(gs[2]),plt.subplot(gs[3]),plt.subplot(gs[4]),plt.subplot(gs[5]),
       plt.subplot(gs[6]),plt.subplot(gs[7]),plt.subplot(gs[8]),plt.subplot(gs[9]),plt.subplot(gs[10]),plt.subplot(gs[11])]
plt.ion()
all_mats = np.zeros((num_fingers,num_fingers,len(SUBJ_IDS)))
for idx,subj_id in enumerate(SUBJ_IDS):
    fmri_data = moving_average(np.load('data/'+subj_id+'_localizer.npy'))[TRS_TO_EXTRACT,:]
    press_data = pd.read_csv('data/'+subj_id+'_localizer_press.csv')
    trial_data = press_data.groupby('trial',sort=False).first() 
    trial_labels = trial_data.target_finger
    sorted_labels=trial_labels.sort_values()
    sorted_fmri_data=fmri_data[sorted_labels.index,:]
    corr_mat = np.arctanh(np.corrcoef(sorted_fmri_data))
    np.fill_diagonal(corr_mat,0)
    mini_mat = np.zeros((num_fingers,num_fingers))
    for diag_element in range(num_fingers):
        base_pos = diag_element*trials_per_finger
        final_pos = (diag_element+1)*trials_per_finger
        mini_mat[diag_element,diag_element]=np.mean(
            corr_mat[base_pos:final_pos,base_pos:final_pos][np.tril_indices(trials_per_finger,-1)])
    for off_diag in zip(np.tril_indices(num_fingers,-1)[0],np.tril_indices(num_fingers,-1)[1]):
        x_base_pos = off_diag[0]*trials_per_finger
        x_final_pos = (off_diag[0]+1)*trials_per_finger
        y_base_pos = off_diag[1]*trials_per_finger
        y_final_pos = (off_diag[1]+1)*trials_per_finger
        mini_mat[off_diag] = np.mean(corr_mat[x_base_pos:x_final_pos,y_base_pos:y_final_pos])
    all_mats[:,:,idx] = mini_mat
    if idx == len(SUBJ_IDS)-1:
        cbar_bool = True
    else:
        cbar_bool = False
    sea.heatmap(mini_mat,ax=axs[idx],vmin=-0.25,vmax=0.25,square=True,cbar=cbar_bool,cmap='coolwarm')
