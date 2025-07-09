# Code used in order to clean the datasets attached in the Mendeley dataset
## By Kenzo Hubert

# ----------------------------------------------------------------
# VOCABULARY:
# - Pauli Test: A personality test commonly given to employees to test their focus and learning rate, derived from the Kraepelin test
#   (https://www.algobash.com/en/types-of-psychological-tests/)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# KEY THINGS TO NOTE:
# - 5 subjects with 3 experimental conditions:
#    - rec_01: User watches posts that they like
#    - rec_02: User taking Pauli test without any distractions
#    - rec_03: User taking Pauli tests with autitory social media distractions
# - Each subject is exposed to all 3 experimental conditions
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# Coding 

## Packages
import pandas as pd
import numpy as np

## Exploratory Data Analysis
df = pd.read_csv('Mendeley/Subject_01/rec01_subject_01.csv')
df.head()
print(df.columns())
print(df.shape())


