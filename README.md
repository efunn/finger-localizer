# finger-localizer
This repository contains sample fMRI and behavioral analyses for a finger pressing task.

## participants
There are 12 participants in this dataset with subject IDs: 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14. Please note that datasets do not exist for subjects 5 and 9, as they dropped out of the experiment before collecting a full finger localizer dataset.

## data files
There are 4 data files per participant:

`ftXXX_localizer.npy` contains the fMRI data for the entire experiment, masked within the hand area (the union of the grey matter mask of M1+S1 and . Each run contains 170 fMRI volumes (TR=2s). The first 20s (10 volumes) are baseline, and each trial after that consists of 2s cue, 10s press, 1s wait, 2s feedback, 1s wait (16s per trial total; 20 trials per run). The dimensions of the dataset are `n_volumes*n_voxels`. These files can be loaded using numpy:

```
import numpy as np
subject_id = 'ft001' # ft002, ft003, etc.
ft001_data = np.load('data/'+subject_id+'_localizer.npy')
```

`ftXXX_localizer_press.csv` contains the pressing data for each participant's localizer session. Each press is a separate row in this file. Labels and run numbers for each trial can be extracting using pandas:

```
import pandas as pd
subject_id = 'ft001' # ft002, ft003, etc.
press_data = pd.read_csv('data/'+subject_id+'_localizer_press.csv')
trial_data = press_data.groupby('trial',sort=False).first() 
trial_labels = trial_data.target_finger
trial_runs = trial_data.run
```

`ftXXX_rrt_pre.csv` and `ftXXX_rrt_post.csv` contains the pressing data for a rapid reaction time (RRT) test, both before (`pre`) and after (`post`) the localizer fMRI session. 

`ftXXX_toj.csv` contains the behavioral data for a temporal order judgment (TOJ) task that occurred in a separate session preceding the localizer fMRI session. 
