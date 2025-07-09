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
#
# - Headband columns that begin with HSI (Horse Shoe Indicator) contain data values of either 1, 2, or 4
#    - 1: Good readings from the specified sensor
#    - 2: Medium readings from the specified sensor
#    - 4: Bad readings from the specified sensor
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# CODING

## Packages
import pandas as pd
import os

## Reading in dataset (singular)
os.chdir('dataset/Mendeley')
df = pd.read_csv('Subject_01/rec01_subject_01.csv')

## Exploratory Data Analysis
""" 
print(df.head())
print(df.columns)
print(df.shape)

print(df.value_counts('Battery', dropna = False))
print(df.value_counts('HSI_TP9', dropna = False)) 
"""

## Creating time-based indices using the `TimeStamp` column
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
df.set_index('TimeStamp', inplace = True)


## Removing observations with no data, and observations no change compared to previous observations

### First step is to remove columns that get in the way of this process:
### The `RAW_{}` columns, `AUX_RIGHT` column, and `Element` columns all contain meaningless changing 
### values that inhibit the measuring of consistency between other relevant recorded values
relevant_columns = [col for col in df.columns]
for i in df.columns:
    if "RAW" in i or "AUX" in i:
        relevant_columns.remove(i)
relevant_columns.remove("Elements")

### Remove all rows that do not fit the following criteria:
###    - The entries in this row are all different from the previous rows (row is unique)
###    - The majority of entries contain values, headband is on the head (row is meaningful)
###    - The entries are all taken with "good" readings from the sensors (row is accurate)
df.drop_duplicates(subset = relevant_columns, inplace = True)
df.dropna(subset = relevant_columns, inplace = True)
df = df[(df["HeadBandOn"] == 1) & (df["HSI_TP9"] == 1) & (df["HSI_AF7"] == 1) & (df["HSI_AF8"] == 1) & (df["HSI_TP10"] == 1)]





## Create a function with the above code that takes in a file path, returning a "cleaned" dataset according to the above criteria
def muse_clean(subject_num: int, record_num: int) -> None:
    """
    Inputs: The subject number (1-5) and the record number (1-3)
    Output: A file output (.csv) that returns a "cleaned" version of the raw data file

    Through entering the requested information (subject number, record number), users are able to download a "cleaned" version of
    the dataset inputted according to the following criteria:

      - The entries in this row are all different from the previous rows (row is unique)
      - The majority of entries contain values, headband is on the head (row is meaningful)
      - The entries are all taken with "good" readings from the sensors (row is accurate)    
    """

    ## Change directory to read in dataset
    os.chdir('dataset/Mendeley')
    df = pd.read_csv(f'Subject_0{subject_num}/rec0{record_num}_subject_0{subject_num}.csv')

    ## Creating time-based indices using the `TimeStamp` column
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.set_index('TimeStamp', inplace = True)


    ## Removing observations with no data, and observations no change compared to previous observations

    ### First step is to remove columns that get in the way of this process:
    ### The `RAW_{}` columns, `AUX_RIGHT` column, and `Element` columns all contain meaningless changing 
    ### values that inhibit the measuring of consistency between other relevant recorded values
    relevant_columns = [col for col in df.columns]
    for i in df.columns:
        if "RAW" in i or "AUX" in i:
            relevant_columns.remove(i)
    relevant_columns.remove("Elements")

    ### Remove all rows that do not fit the following criteria:
    ###    - The entries in this row are all different from the previous rows (row is unique)
    ###    - The majority of entries contain values, headband is on the head (row is meaningful)
    ###    - The entries are all taken with "good" readings from the sensors (row is accurate)
    df.drop_duplicates(subset = relevant_columns, inplace = True)
    df.dropna(subset = relevant_columns, inplace = True)
    df = df[(df["HeadBandOn"] == 1) & (df["HSI_TP9"] == 1) & (df["HSI_AF7"] == 1) & (df["HSI_AF8"] == 1) & (df["HSI_TP10"] == 1)]

    ## Download dataset into current working directory (Mendeley)
    df.to_csv(f"rec0{record_num}_subject0{subject_num}_cleaned.csv")



