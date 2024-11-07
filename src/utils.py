import seaborn
import pandas as pd
import numpy as np
import random
import string
from pathlib import Path
# pd.set_option('future.no_silent_downcasting', True) # for future compatibility of .replace method

def inter_len(set1, set2, dfs=False):
    """
    Compute the intersection of two sets and return ratio.
    """
    if dfs:     
        n = len(set1.index.intersection(set2.index))
        p = float(n)/max(len(set1.index), len(set2.index))
    else:
        n = len(set1.intersection(set2))
        p = float(n)/max(len(set1), len(set2))
    return (n, p)

def plot_pairwise_intersection_ratio(dfs, tech_names):
    """
    Creates heatmap with pairwise intersection ratios, i.e., how many indices of different 
    dataframes are the same. 
    """
    n = len(dfs)
    overlaps = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            overlaps[i][j] = inter_len(dfs[i].index, dfs[j].index)[1]

    seaborn.heatmap(overlaps, cmap='hot', xticklabels = tech_names+["ids"], yticklabels = tech_names+["ids"])

def join_selected_dfs(dfs, df_names, df_names_to_join):
    """ 
    Join list of dfs.
    """
    dfs_ids = [df_names.index(df_name) for df_name in df_names_to_join]
    result = pd.DataFrame()

    for i in dfs_ids:
        if result.empty:
            result = dfs[i].copy()
        else:
            result = result.join(dfs[i], how="inner")

    return result

def add_prefix_to_cols(df, prefix):
    """ 
    Adds prefix to colnames.
    """
    df.columns = [prefix+col for col in df.columns]

def generate_sample_id(characters, length=10):
    """ 
    Generates random sample id. Same style of id as used in medical datasets. 
    """
    rand_string = ''.join(random.choices(characters, k=length))

    return rand_string[:length//2]+'-'+rand_string[length//2:]

def has_duplicates(seq):
    return len(seq) != len(set(seq))

def gen_unique_ids(n):
    """ 
    Generate unique list of random ids for a medical dataframe.
    """
    length = 10
    characters = string.ascii_lowercase
         
    ids = [generate_sample_id(characters, length) for i in range(n)]
    assert(not has_duplicates(ids))
    return ids

def prepare_and_save_data(df: pd.DataFrame, 
                          directory_path_: str, 
                          file_name: str, 
                          treatment_col: str ='tre_nivolumab', 
                          directory_name: str ="", 
                          counterfactual_outcomes = None):
    """ 
    Brings data into the format required for the Tumor board and creates directories where necessary.
    """
    # Get colnames
    outcome_cols = [col for col in df.columns if col.startswith("out")]

    # Create directories
    prepared_path_ = directory_path_ + file_name[:-4] + '/' + directory_name + '/' + treatment_col + '/'

    #Return to this if errors occur
    #prepared_path_ = directory_path_ + file_name[:-4] + directory_name + '/' + treatment_col + '/'

    outcomes_path_ = prepared_path_+'outcomes/'
    treatment_path_ = prepared_path_+'treatment/'
    features_path_ = prepared_path_+'features/'

    # Create directory and subdirectories 
    Path(prepared_path_).mkdir(parents=True, exist_ok=True)
    Path(outcomes_path_).mkdir(parents=True, exist_ok=True)
    Path(treatment_path_).mkdir(parents=True, exist_ok=True)
    Path(features_path_).mkdir(parents=True, exist_ok=True)
    
    # Create treatment file
    df_treatment_factual = pd.get_dummies(df[treatment_col]).replace({False:0, True:1})
    df_treatment_factual.columns = ['a'+str(i)+'_'+str(name) for i,name in enumerate(df_treatment_factual.columns)]
    df_treatment_factual.to_csv(treatment_path_ + 'treatment_factual.csv', index_label='ID')

    if counterfactual_outcomes is not None:
        df_treatment_data = df_treatment_factual.replace({0:1})
        df_treatment_data.to_csv(treatment_path_ + 'treatment_data.csv', index_label='ID')
        counterfactual_outcomes.to_csv(outcomes_path_ + 'outcomes_data.csv', index_label='ID')

    # Create outcome file
    df_outcomes_factual = df[outcome_cols]
    df_outcomes_factual.columns = ['y'+str(i)+'_'+str(name) for i,name in enumerate(df_outcomes_factual.columns)]
    df_outcomes_factual.to_csv(outcomes_path_ + 'outcomes_factual.csv', index_label='ID')

    # Find out how many technologies are used
    feature_names = df.columns[df.columns.str.match(r'^\d{2}_')]
    tech_numbers = set([feature[:2] for feature in feature_names])

    # Create features files
    df_features = pd.DataFrame()
    for tech in tech_numbers:
        df_feature_factual = df[df.columns[df.columns.str.startswith(tech)]]
        df_feature_factual.columns = ['p'+str(int(tech))+'_'+str(i)+'_'+tech_name[3:] for i, tech_name in enumerate(df_feature_factual.columns)]
        df_features = pd.concat([df_features, df_feature_factual], axis=1)
        df_feature_factual.to_csv(features_path_ + 'patient_data_'+str(int(tech))+'.csv', index_label='ID')
    df_features.to_csv(features_path_ + 'patient_data_all.csv', index_label='ID')

    # Return prepared path
    return prepared_path_