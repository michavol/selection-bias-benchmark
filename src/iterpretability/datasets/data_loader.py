import pickle
import pandas as pd
import numpy as np
from catenets.datasets import load as catenets_load
from src.iterpretability.datasets.news.process_news import process_news
from src.iterpretability.datasets.tcga.process_tcga import process_tcga
from src.utils import gen_unique_ids
import os

def remove_unbalanced(df, threshold=0.8):
    ub_col = []
    for col in df.select_dtypes(exclude='object').columns:
        if df[col].nunique() < 6:
            distribution = df[col].value_counts(normalize=True)
        
            if distribution.max() > threshold:
                ub_col.append(col)

    df = df.drop(columns=ub_col)
    return df

def load(dataset_name: str, train_ratio: float = 1.0, 
         directory_path_: str = None, 
         repo_path: str = None,
         debug=False,
         sim_type=None,
         n_samples=None):
    """
    Load the dataset
    """
    feature_names = None

    if "tcga" in dataset_name:
        try:
            tcga_dataset = pickle.load(
                open(
                    repo_path + "/data/tcga/" + str(dataset_name) + ".p",
                    "rb",
                )
            )
        except:
            process_tcga(
                max_num_genes=100, file_location=repo_path + "/data/tcga/"
            )
            tcga_dataset = pickle.load(
                open(
                    repo_path + "/data/tcga/" + str(dataset_name) + ".p",
                    "rb",
                )
            )
        # get gene names
        # ids = tcga_dataset["ids"]
        X_raw = tcga_dataset["rnaseq"][:5000]
        
    elif "news" in dataset_name:
        try:
            news_dataset = pickle.load(
                open(
                    "src/iterpretability/datasets/news/" + str(dataset_name) + ".p",
                    "rb",
                )
            )
        except:
            process_news(
                max_num_features=100, file_location="src/iterpretability/datasets/news/"
            )
            news_dataset = pickle.load(
                open(
                    "src/iterpretability/datasets/news/" + str(dataset_name) + ".p",
                    "rb",
                )
            )
        X_raw = news_dataset

    elif "twins" in dataset_name:
        # Total features  = 39
        X_raw, _, _, _, _, _ = catenets_load(dataset_name, train_ratio=1.0)

        # Remove columns which almost only contain one value from X_raw
        # Assuming X_raw is your numpy array
        # threshold = 2  # Adjust this value based on your definition of "almost only one value"

        # # Find the columns to keep
        # cols_to_keep = [i for i in range(X_raw.shape[1]) if np.unique(X_raw[:, i]).size > threshold and i != 3]

        # # Create a new array with only the columns to keep
        # X_raw = X_raw[:, cols_to_keep]
        
    elif "acic" in dataset_name:
        # Total features  = 55
        X_raw, _, _, _, _, _, _, _ = catenets_load("acic2016")

    elif dataset_name.startswith("depmap_drug_screen"):
        data = pd.read_csv(repo_path + "/data/DepMap_24Q2/real/"+dataset_name+".csv", index_col=0)
        outcomes = data[["LFC_az_628", "LFC_imatinib"]].to_numpy()
        outcomes = outcomes.reshape(outcomes.shape[0], outcomes.shape[1], 1)
        data = data.drop(["LFC_az_628", "LFC_imatinib"], axis=1)
        X_raw = data.to_numpy()

    elif dataset_name.startswith("depmap_crispr_screen"):
        data = pd.read_csv(repo_path + "/data/DepMap_24Q2/real/"+dataset_name+".csv", index_col=0)
        outcomes = data[["LFC_BRAF", "LFC_EGFR"]].to_numpy()
        outcomes = outcomes.reshape(outcomes.shape[0], outcomes.shape[1], 1)
        data = data.drop(["LFC_BRAF", "LFC_EGFR"], axis=1)
        X_raw = data.to_numpy()

    elif dataset_name.startswith("ovarian_semi_synthetic"):
        data = pd.read_csv(repo_path + "/data/DepMap_24Q2/real/"+dataset_name+".csv", index_col=0)
        outcomes = data[["pred_a0_y0", "pred_a1_y0"]].to_numpy()
        outcomes = outcomes.reshape(outcomes.shape[0], outcomes.shape[1], 1)
        data = data.drop(["pred_a0_y0", "pred_a1_y0"], axis=1)
        X_raw = data.to_numpy()

    elif dataset_name.startswith("melanoma_semi_synthetic"):
        data = pd.read_csv(repo_path + "/data/DepMap_24Q2/real/"+dataset_name+".csv", index_col=0)
        outcomes = data[["pred_a0_y2 (immuno)", "pred_a1_y2 (immuno)"]].to_numpy()
        outcomes = outcomes.reshape(outcomes.shape[0], outcomes.shape[1], 1)
        outcomes = outcomes.astype(int)
        data = data.drop(["pred_a0_y2 (immuno)", "pred_a1_y2 (immuno)"], axis=1)
        X_raw = data.to_numpy()

    elif dataset_name.startswith("toy_data"):
        if sim_type == "T":
            raise ValueError("Toy example data does not have treatment outcomes")
        data = pd.read_csv(repo_path + "/data/toy/"+dataset_name+".csv", index_col=0)
        X_raw = data.to_numpy()

    elif dataset_name.startswith("cytof"):
        data = pd.read_csv(directory_path_ + dataset_name + ".csv", index_col=0)

        feature_names = data.columns 

        # Also return true outcomes for TSimulation
        if sim_type == "T":
            all_treatment_outcomes_cols = [col for col in data.columns if col.startswith("09")]
            outcomes = data[all_treatment_outcomes_cols[:2]] 
            data = data.drop(all_treatment_outcomes_cols, axis=1)
          
            # Raise value error if outcomes contains string or nan values
            if outcomes.isnull().values.any():
                raise ValueError("Outcomes contain nan values")
            if outcomes.dtypes.apply(lambda x: x == 'object').any():
                raise ValueError("Outcomes contain string values")
            
            if type(outcomes) == pd.DataFrame:
                outcomes = outcomes.replace({False:0, True:1})
                outcomes = outcomes.to_numpy()
                outcomes = outcomes.reshape(outcomes.shape[0], outcomes.shape[1], 1)

        X_raw = data


    elif directory_path_ is not None and os.path.isfile(directory_path_ + dataset_name + ".csv"):
        data = pd.read_csv(directory_path_ + dataset_name + ".csv", index_col=0)

        all_treatment_outcomes_cols = [col for col in data.columns if col.startswith("09")][:5]

        # Also return true outcomes for TSimulation
        if sim_type == "T":
            outcomes = data[all_treatment_outcomes_cols]

            # Raise value error if outcomes contains string or nan values
            if outcomes.isnull().values.any():
                raise ValueError("Outcomes contain nan values")
            if outcomes.dtypes.apply(lambda x: x == 'object').any():
                raise ValueError("Outcomes contain string values")
            
            if type(outcomes) == pd.DataFrame:
                outcomes = outcomes.replace({False:0, True:1})
                outcomes = outcomes.to_numpy()
                outcomes = outcomes.reshape(outcomes.shape[0], outcomes.shape[1], 1)

        # Drop all columns starting with tre or out 
        X_raw = data.loc[:, ~data.columns.str.startswith('tre')]
        X_raw = X_raw.loc[:, ~X_raw.columns.str.startswith('out')]

        # One hot encode all categorical features
        X_raw = pd.get_dummies(X_raw)

        # Make sure there are no boolean variables
        X_raw = X_raw.replace({False:0, True:1})

        # Remove highly unbalanced features
        X_raw = remove_unbalanced(X_raw, 0.8)

        # Normalize all columns of the dataframe
        X_raw = X_raw.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # X_raw.index = gen_unique_ids(X_raw.shape[0])

    else:
        raise Exception("Unknown dataset " + str(dataset_name) + "File:", directory_path_ + dataset_name + ".csv")

    debug_size = 200
    if type(X_raw) == pd.DataFrame:
        X_raw = X_raw.to_numpy()
    
    if train_ratio == 1.0:
        if debug:
            X_raw = X_raw[:int(debug_size*train_ratio),:]
            if sim_type == "T":
                outcomes = outcomes[:int(debug_size*train_ratio),:]

        if sim_type == "T":
            return X_raw, outcomes, feature_names
        else:
            return X_raw, feature_names
    
    else:
        X_raw_train = X_raw[: int(train_ratio * X_raw.shape[0])]
        X_raw_test = X_raw[int(train_ratio * X_raw.shape[0]) :]

        if debug:
            X_raw_train = X_raw_train[:int(debug_size*train_ratio),:]
            X_raw_test = X_raw_test[:int(debug_size*(1-train_ratio)),:]
            
            if sim_type == "T":
                outcomes_train = outcomes[:int(debug_size*train_ratio),:]
                outcomes_test = outcomes[int(debug_size*train_ratio):int(debug_size*(1-train_ratio)),:]

        if sim_type == "T":
            return X_raw_train, X_raw_test, outcomes_train, outcomes_test, feature_names
        else:
            return X_raw_train, X_raw_test, feature_names
