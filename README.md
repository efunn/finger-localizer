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

## pattern overlap analysis

A basic version of finger pattern overlap analysis is found in `analysis.py`. Similar to a decoding analysis, we extract the patterns for each finger (40 samples for each finger). For each finger, we then submit the patterns to a 1-sample t-test to determine which voxels have activity that appears significantly greater or less than baseline levels. To determine pattern overlap, we threshold these statistics (in this sample, an arbitrary `PVAL_THRESH = 1e-4` is used), and then calculate the DICE coefficient between the thresholded statistical maps of adjacent finger pairs (a value between 0 and 1, where 0 is zero overlap and 1 is complete overlap).

This basic version has a couple issues. First, patterns are estimated for each finger independently. We should be able to fit a model for all fingers simultaneously, rather than simply looking if activity is below or above baseline for each finger independently. Also, we are only looking at the 40 discrete patterns for each finger, whereas we could be fitting a statistical model to the entire timecourse of the experiment.

We can use a [generalized linear model (GLM)](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html#statsmodels.genmod.generalized_linear_model.GLM) to perform a more accurate stastical estimation of the patterns. The python `statsmodels` GLM takes 2 main arguments, which they call the endogenous response (`endog=`) and exogenous variables (`exog=`). In this case, `endog` will be the fMRI data and `exog` will be the experimental design. Another name for `exog` is a `design matrix`, which you will need to construct from the trial data. You'll need to create a `n_TR*n_fingers` matrix that describes how much activity you expect from each finger at each time point. In practice, you'll be creating a matrix with 4 columns, where the value at each time point will either be 0 (finger not being pressed at that time) or 1 (finger being pressed at that time). Then, you'll need to convolve each column of the design matrix with the expected fMRI response, known as the [hemodynamic response](https://en.wikipedia.org/wiki/Haemodynamic_response). You'll find a function `gen_hrf()` in `analysis.py` that will create this for you. Once you've created your convolved design matrix, you can fit the GLM using `GLM.fit()` and then extract model parameters and p-values from that model.
