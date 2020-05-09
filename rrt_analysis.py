import numpy as np
import pandas as pd

NUM_FINGERS = 4
NUM_FINGERS_2 = 5
SUBJ_NUMS = [1,2,3,4,6,7,8,10,11,12,13,14]
# SUBJ_NUMS = [1,2,4,6,7,10,11,12,13,14] # fingtrain subjs only
SUBJ_IDS = ['ft'+str(subj_num).zfill(3) for subj_num in SUBJ_NUMS]

SUBJ_NUMS_2 = [1,2,3,4,5,6,7,8,9,10]
SUBJ_IDS_2 = ['reward-srt/subj'+str(subj_num).zfill(3) for subj_num in SUBJ_NUMS_2]

def calculate_presses(subj_id, num_fingers=4):
    press_df = pd.read_csv('data/'+subj_id+'_rrt_post.csv')
    press_df = press_df[press_df.selected_finger>-1].reset_index()
    press_results = np.zeros((num_fingers,num_fingers))
    for target_finger in range(num_fingers):
        for selected_finger in range(num_fingers):
            press_results[target_finger,selected_finger] = (
                len(press_df[press_df.target_finger==target_finger][press_df.selected_finger==selected_finger]))
    return press_results

def calculate_presses_2(subj_id, num_fingers=5):
    press_df = pd.read_csv('data/'+subj_id+'/data.txt')
    press_df = press_df[press_df.press>-1].reset_index()
    press_results = np.zeros((num_fingers,num_fingers))
    for target_finger in range(num_fingers):
        for selected_finger in range(num_fingers):
            press_results[target_finger,selected_finger] = (
                len(press_df[press_df.probe==target_finger][press_df.press==selected_finger]))
    return press_results

press_df = np.zeros((NUM_FINGERS,NUM_FINGERS,len(SUBJ_NUMS)))
for subj_idx, subj_id in enumerate(SUBJ_IDS):
    press_df[:,:,subj_idx] = calculate_presses(subj_id)

press_df_2 = np.zeros((NUM_FINGERS_2,NUM_FINGERS_2,len(SUBJ_NUMS_2)))
for subj_idx, subj_id in enumerate(SUBJ_IDS_2):
    press_df_2[:,:,subj_idx] = calculate_presses_2(subj_id)

