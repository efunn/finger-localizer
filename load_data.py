import numpy as np
import pandas as pd

SUBJ_NUMS = [1,2,3,4,6,7,8,10,11,12,13,14]
SUBJ_IDS = ['ft'+str(subj_num).zfill(3) for subj_num in SUBJ_NUMS]

# sample for loading one subject
subject_id = SUBJ_IDS[0]
fmri_data = np.load('data/'+subject_id+'_localizer.npy')
press_data = pd.read_csv('data/'+subject_id+'_localizer_press.csv')
trial_data = press_data.groupby('trial',sort=False).first() 
trial_labels = trial_data.target_finger
trial_runs = trial_data.run
