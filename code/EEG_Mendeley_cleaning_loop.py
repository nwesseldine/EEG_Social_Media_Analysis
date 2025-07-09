# A for loop that incorporates the code from the `EEG_Mendeley_cleaning.py` file to produce cleaned datasets
from EEG_Mendeley_cleaning import muse_clean

for patient in range(1, 6):
    for experiment in range(1, 4):
        muse_clean(patient, experiment)