import argparse
import pandas as pd
import torch

import os
# import pyro
import numpy as np
print(os.getcwd())

import pandas as pd

from sklearn import preprocessing



def load_data(path_data, num_sheet):
    cf_path = path_data + 'counterfactuals/' + str(num_sheet) + '_cf' + '.csv'
    f_path = path_data + 'factuals/'+ str(num_sheet) + '.csv'
    # print(cf_path)
    # print(f_path)
    # append the CSV files
    df_cf = pd.read_csv(cf_path)
    # print(df_cf)
    df_f = pd.read_csv(f_path)
    # print(df_f)
    x_path = path_data + 'x.csv'
    df_x = pd.read_csv(x_path)
    
    
    x_raw = pd.read_csv(x_path, header=0, sep=',')
    tymu_table_cf = df_cf.to_numpy()[:, 1:]
    tymu_table_f = df_f.to_numpy()[:, 1:]
    # print(tymu_table_cf.shape) 
    # print(tymu_table_cf)
    # print(tymu_table_f.shape) 
    
#     print(df_cf['sample_id'].values)
    
    sample_id_list = df_cf['sample_id'].values

    # Filter the DataFrame to get rows where sample_id matches the values in your list
    filtered_df = x_raw[x_raw['sample_id'].isin(sample_id_list)]
#     print(filtered_df)
    x_table = filtered_df.to_numpy()[:, 1:]
    # print(x_table.shape) 
    # print(x_table)
    
    
    cov_scalar = preprocessing.StandardScaler()
    y_out_scaler = preprocessing.StandardScaler()

    x = cov_scalar.fit_transform(x_table)
    # print(np.min(x), np.max(x))
    y_out_scaler.fit(np.concatenate([df_cf['y0'].values, df_cf['y1'].values]).reshape(-1, 1))
    y_0 = y_out_scaler.transform(df_cf['y0'].values.reshape(-1, 1))
    y_1 = y_out_scaler.transform(df_cf['y1'].values.reshape(-1, 1))


    mu_out_scaler = preprocessing.StandardScaler()

    mu_out_scaler.fit(np.concatenate([df_cf['y0'].values, df_cf['y1'].values]).reshape(-1, 1))
    mu_0 = mu_out_scaler.transform(df_cf['y0'].values.reshape(-1, 1))
    mu_1 = mu_out_scaler.transform(df_cf['y1'].values.reshape(-1, 1))
    
    t = df_f['z'].values.reshape(-1, 1)
    full_data = np.concatenate((t,y_0,y_1,mu_0,mu_1, x) , axis=1)
    # print(full_data.shape) #(4802, 87)
    
    
#     print('Start normalizing data.')
    norm_data_df = pd.DataFrame(full_data)


    # save
    norm_data_folder = './acic2018_norm_data/'
    if not os.path.exists(norm_data_folder):
        os.makedirs(norm_data_folder)

    norm_data_path = norm_data_folder + num_sheet + '.csv'
    norm_data_df.to_csv(norm_data_path, index=False)
    # print(norm_data_path)
    
    full_table = full_data

    mask = np.ones(full_table.shape)
    # print(mask.shape)
    mask[:, 3] = 0 # mask mu0
    mask[:, 4] = 0 # mask mu1


    for i in range(full_table.shape[0]):
        t = full_table[i, 0]
        if t == 0:
            mask[i, 2] = 0 # mask y1

        if t == 1:
            mask[i, 1] = 0 # mask y0

#     print('finish generating mask.')
    #print(mask)

    mask_df = pd.DataFrame(mask)

    # save
    mask_folder = './acic2018_mask/'
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    mask_path = mask_folder + num_sheet + '.csv'
    mask_df.to_csv(mask_path, index=False)
    # print(mask_path)
    
#     print('Finish generating mask.')
    # print('============== Finish this dataset')

    
