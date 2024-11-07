import pickle
import yaml
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def process_func(path: str, aug_rate=1, missing_ratio=0.1, train=True, dataset_name = 'acic', current_id='0', data_path=None, x_dim=None):
    # data = pd.read_csv(path, sep = ',', decimal = ',', skiprows=[0])
    data = pd.read_csv(path, sep = ',', decimal = ',')
    data.replace("?", np.nan, inplace=True)
    data_aug = pd.concat([data] * aug_rate)

    observed_values = data_aug.values.astype("float32")
    observed_masks = ~np.isnan(observed_values)

    masks = observed_masks.copy()

    if dataset_name == 'acic2016':

        load_mask_path = "./data_acic2016/acic2016_mask/" + current_id + ".csv"
        #print(load_mask_path)


    # ========================
    # acic2018
    if dataset_name == 'acic2018':
        # load_mask_path = "./data_acic2018/acic2018_mask/00ea30e866f141d9880d5824a361a76a.csv"
        load_mask_path = data_path + "acic2018_mask/" + current_id + ".csv"
        #print(load_mask_path)

    if dataset_name == 'depmap':
        load_mask_path = "./data_depmap/depmap_mask/" + current_id + ".csv"
        #print(load_mask_path)

    load_mask = pd.read_csv(load_mask_path, sep = ',', decimal = ',')
    load_mask = load_mask.values.astype("float32")

    if train:
        # for each column, mask {missing_ratio} % of observed values.
        # for col in range(observed_values.shape[1]):  # col #
        #     obs_indices = np.where(masks[:, col])[0]
        #     miss_indices = np.random.choice(
        #         obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        #     )
        #     masks[miss_indices, col] = False
        # # gt_mask: 0 for missing elements and manully maksed elements
        # gt_masks = masks.reshape(observed_masks.shape)
            
        gt_masks = load_mask # get load mask for training # no ycf, mu0, mu1

        observed_values = np.nan_to_num(observed_values)
      
        observed_masks = observed_masks.astype(int)
        gt_masks = gt_masks.astype(int)

    else:

        gt_masks = load_mask # get load mask for training # no ycf, mu0, mu1
        # no yf for testing
        gt_masks[:, 1] = 0 # mask y0
        gt_masks[:, 2] = 0 # mask y1
        

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype(int)
        gt_masks = gt_masks.astype(int)


    return observed_values, observed_masks, gt_masks


class tabular_dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(
        self, eval_length=100, use_index_list=None, aug_rate=1, missing_ratio=0.1, seed=0, train=True, dataset_name = 'acic', current_id='0', data_path=None, x_dim=None
    ):  
        if dataset_name == 'acic2016':
            self.eval_length = 87
        if dataset_name == 'acic2018':
            #print(x_dim)
            self.eval_length = x_dim + 4 # 189 #182
            
        if dataset_name == 'depmap':
            self.eval_length = 189 

        np.random.seed(seed)

        if dataset_name == 'acic2016':

            dataset_path = "./data_acic2016/acic2016_norm_data/" + current_id + ".csv"

            #print('dataset_path', dataset_path)

            processed_data_path = (
                f"./data_acic2016/missing_ratio-{missing_ratio}_seed-{seed}.pk"
            ) # modify the processed data path
            processed_data_path_norm = (
                f"./data_acic2016/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
            )
            #self.observed_values, self.observed_masks, self.gt_masks = None, None, None
            
            os.system('rm {}'.format(processed_data_path))

        # ========================
        # acic2018
        if dataset_name == 'acic2018':
            # dataset_path = "./data_acic2018/acic2018_norm_data/00ea30e866f141d9880d5824a361a76a.csv"
            # dataset_path = "./data_acic2018/acic2018_norm_data/194fe0e3c1644d41a5085b92d2fe7e54.csv"
            dataset_path = data_path + "acic2018_mask/" + current_id + ".csv"

            #print('dataset_path', dataset_path)

            processed_data_path = (
                data_path + "missing_ratio-{missing_ratio}_seed-{seed}.pk"
            ) # modify the processed data path
            processed_data_path_norm = (
                data_path + "missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
            )
            #self.observed_values, self.observed_masks, self.gt_masks = None, None, None
            
            os.system('rm {}'.format(processed_data_path))

        if dataset_name == 'depmap':
            dataset_path = "./data_depmap/depmap_norm_data/" + current_id + ".csv"

            #print('dataset_path', dataset_path)

            processed_data_path = (
                f"./data_depmap/missing_ratio-{missing_ratio}_seed-{seed}.pk"
            )
            processed_data_path_norm = (
                f"./data_depmap/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
            )

        if not os.path.isfile(processed_data_path):
            self.observed_values, self.observed_masks, self.gt_masks = process_func(
                dataset_path, aug_rate=aug_rate, missing_ratio=missing_ratio, train=train, dataset_name=dataset_name, current_id=current_id, data_path=data_path
            )

            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
            #print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):
            with open(processed_data_path_norm, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
            #print("--------Normalized dataset loaded--------")

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list


    def __getitem__(self, org_index):
        index = self.use_index_list[org_index] #672
        s = {
            "observed_data": self.observed_values[index], # (30,)
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.1, dataset_name = 'acic2018', current_id='0', training_size=1, data_path=None, x_dim=None):
    #print(x_dim)
    dataset = tabular_dataset(missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, current_id=current_id, data_path=data_path, x_dim=x_dim)
    #print(f"Dataset size:{len(dataset)} entries") #  Dataset size: 747 entries

    indlist = np.arange(len(dataset))
    train_size = training_size
    tsi = int(len(dataset) * train_size)
    #print('test start index', tsi)
    if tsi % 8 == 1 or int(len(dataset) * 0.2) % 8 == 1:
        tsi = tsi + 3


    if dataset_name == 'acic2016':
        test_index = indlist[3842:]
        remain_index = np.arange(0, 3842)

        np.random.shuffle(remain_index)
        num_train = (int)(len(remain_index) * 1)

        train_index = remain_index[: 3842] 
        valid_index = remain_index[: 384] 

        # Here we perform max-min normalization.
        processed_data_path_norm = (
            f"./data_acic2016/missing_ratio-{missing_ratio}_seed-{seed}_current_id-{current_id}_max-min_norm.pk"
        )
        # if not os.path.isfile(processed_data_path_norm):
        # normalize anyway.
        #print(
        #     "------------- Perform data normalization and store the mean value of each column.--------------"
        # )
        # data transformation after train-test split.

    if dataset_name == 'acic2018':
        test_index = indlist[tsi:]
        remain_index = np.arange(0, tsi)

        np.random.shuffle(remain_index)
        num_train = (int)(len(remain_index) * 1)

        train_index = remain_index[: tsi] 
        valid_index = remain_index[: int(tsi*0.1)] 

        # Here we perform max-min normalization.
        processed_data_path_norm = (
            data_path + f"missing_ratio-{missing_ratio}_seed-{seed}_current_id-{current_id}_max-min_norm.pk"
        )
        # if not os.path.isfile(processed_data_path_norm):
        # normalize anyway.
        # print(
        #     "------------- Perform data normalization and store the mean value of each column.--------------"
        # )
        # data transformation after train-test split.

    if dataset_name == 'depmap':
        test_index = indlist[tsi:]
        remain_index = np.arange(0, tsi)

        np.random.shuffle(remain_index)
        num_train = (int)(len(remain_index) * 1)

        train_index = remain_index[: tsi] 
        valid_index = remain_index[: int(tsi*0.1)] 

        # Here we perform max-min normalization.
        processed_data_path_norm = (
            f"./data_depmap/missing_ratio-{missing_ratio}_seed-{seed}_current_id-{current_id}_max-min_norm.pk"
        )
        # if not os.path.isfile(processed_data_path_norm):
        # normalize anyway.
        # print(
        #     "------------- Perform data normalization and store the mean value of each column.--------------"
        # )
        # data transformation after train-test split.
    
    col_num = dataset.observed_values.shape[1]
    max_arr = np.zeros(col_num)
    min_arr = np.zeros(col_num)
    mean_arr = np.zeros(col_num)
    # for k in range(col_num):
    #     # Using observed_mask to avoid counting missing values.
    #     obs_ind = dataset.observed_masks[train_index, k].astype(bool)
    #     temp = dataset.observed_values[train_index, k]
    #     max_arr[k] = max(temp[obs_ind])
    #     min_arr[k] = min(temp[obs_ind])
    # print(f"--------------Max-value for each column {max_arr}--------------")
    # print(f"--------------Min-value for each column {min_arr}--------------")
    # #======================min-max_normalize=============================
    # # original
    # # dataset.observed_values = (
    # #     (dataset.observed_values - 0 + 1) / (max_arr - 0 + 1)
    # # ) * dataset.observed_masks

    # # min-max normalize
    # dataset.observed_values = (dataset.observed_values - min_arr) / (max_arr - min_arr)
    # print(dataset.observed_values)
    # print('Finish min-max normalization of the input')


    with open(processed_data_path_norm, "wb") as f:
        pickle.dump(
            [dataset.observed_values, dataset.observed_masks, dataset.gt_masks], f
        )

    # Create datasets and corresponding data loaders objects.
    if training_size > 0:
        train_dataset = tabular_dataset(
            use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, current_id=current_id, x_dim=x_dim, data_path=data_path
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

        valid_dataset = tabular_dataset(
            use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, train=False, dataset_name = dataset_name, current_id=current_id, x_dim=x_dim, data_path=data_path
        )
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    else:
        train_dataset = None
        train_loader = None
        valid_dataset = None
        valid_loader = None
    # valid_dataset = None
    # valid_loader = None

    test_dataset = tabular_dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, train=False, dataset_name = dataset_name, current_id=current_id, x_dim=x_dim, data_path=data_path
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    # print(f"Training dataset size: {len(train_dataset)}")
    # # print(f"Validation dataset size: {len(valid_dataset)}")
    # print(f"Testing dataset size: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader
