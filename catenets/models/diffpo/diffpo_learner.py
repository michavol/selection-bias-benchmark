from typing import Any, Callable, List

import numpy as np
import torch
from torch import nn
import os
import tqdm
import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIM_P_OUT,
    DEFAULT_DIM_P_R,
    DEFAULT_DIM_S_OUT,
    DEFAULT_DIM_S_R,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_R,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_PENALTY_ORTHOGONAL,
    DEFAULT_SEED,
    DEFAULT_NJOBS,
    DEFAULT_STEP_SIZE,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.torch.base import DEVICE, BaseCATEEstimator
from catenets.models.torch.utils.model_utils import make_val_split
import pandas as pd
# Hydra
from omegaconf import DictConfig
import json
import datetime

from .src.main_model_table import TabCSDI
from .src.utils_table import train
from .dataset_acic import get_dataloader

from .PropensityNet import load_data


torch.manual_seed(0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DiffPOLearner(BaseCATEEstimator):
    """
    A flexible treatment effect estimator based on the EconML framework.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_features: int,
        binary_y: bool,
    ) -> None:
        self.config = cfg.DiffPOLearner
        self.diffpo_path = cfg.diffpo_path
        self.config.diffusion.cond_dim = num_features+1 # make sure inner dimension matches the dataset
        self.est = None
        self.propnet = None
        self.device = DEVICE
        self.cate_cis = None # confidence intervals, dim: 2, n, num_T-1, dim_Y
        self.pred_outcomes = None

        # create folder if diffpo_path + 'data' does not exist
        if not os.path.exists(self.diffpo_path):
            os.makedirs(self.diffpo_path)
        
        # Store data for their pipeline
        self.data_dir = self.diffpo_path+'/data/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        return None

    def reshape_data(self, X: np.ndarray, w: np.ndarray, outcomes: np.ndarray) -> None:
        data = np.concatenate([w.reshape(-1,1),outcomes[:,0],outcomes[:,1],outcomes[:,0],outcomes[:,1],X], axis=1)
        data_df = pd.DataFrame(data)
        # Create masking array of same shape as pp_data and initialize with 1s
        mask = np.ones(data_df.shape)
        mask[:,1] = w
        mask[:,2] = 1-w
        mask[:,3] = 0
        mask[:,4] = 0
        mask_df = pd.DataFrame(mask)

        return data_df, mask_df

    def train(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, outcomes:np.ndarray) -> None:
        """
        Prepare data and train DiffPO Learner
        """
        log.info("Training data shapes: X: {}, Y: {}, T: {}".format(X.shape, y.shape, w.shape))

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        data, mask = self.reshape_data(X, w, outcomes)
        
        # create destination folders if not exist
        if not os.path.exists(self.data_dir+"acic2018_norm_data/"):
            os.makedirs(self.data_dir+"acic2018_norm_data/")
        if not os.path.exists(self.data_dir+"acic2018_mask/"):
            os.makedirs(self.data_dir+"acic2018_mask/")

        # save intermediate data
        data.to_csv(self.data_dir+"acic2018_norm_data/data_pp.csv", index=False)
        mask.to_csv(self.data_dir+"acic2018_mask/data_pp.csv", index=False)

        # Remove old files
        if os.path.exists(self.data_dir+"missing_ratio-0.2_seed-1_current_id-data_max-min_norm.pk"):
            os.remove(self.data_dir+"missing_ratio-0.2_seed-1_current_id-data_max-min_norm.pk")
        if os.path.exists(self.data_dir+"missing_ratio-0.2_seed-1.pk"):
            os.remove(self.data_dir+"missing_ratio-0.2_seed-1.pk")

        # Create folder
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # define these as variables
        nfold = 1
        config = "acic2018.yaml"
        current_id = "data_pp"
        device = DEVICE
        seed = 1
        testmissingratio = 0.2
        unconditional = 0
        modelfolder = ""
        nsample = 1
        perform_training = 1

        foldername = self.diffpo_path + "/save/acic_fold" + str(nfold) + "_" + current_time + "/"
        # print("model folder:", foldername)
        os.makedirs(foldername, exist_ok=True)

        current_id = "data_pp"
        # print('Start exe_acic on current_id', current_id)

        # Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
        training_size = 1
        
        train_loader, valid_loader, _ = get_dataloader(
            seed=seed,
            nfold=nfold,
            batch_size=self.config["train"]["batch_size"],
            missing_ratio=testmissingratio,
            dataset_name = self.config["dataset"]["data_name"],
            current_id = current_id,
            training_size = training_size,
            data_path=self.data_dir,
            x_dim=X.shape[1],
        )

        #=======================First train and fix propnet======================
        # Train a propensitynet on this dataset

        propnet = load_data(dataset_name = self.config["dataset"]["data_name"], current_id=current_id, x_dim=X.shape[1], data_path=self.data_dir)

        # frozen the trained_propnet
        # print('Finish training propnet and fix the parameters')
        propnet.eval()
        # ========================================================================

        propnet = propnet.to(device)

        model = TabCSDI(self.config, self.device).to(self.device)
        # Train the model
        train(
            model,
            self.config["train"],
            train_loader,
            valid_loader=valid_loader,
            valid_epoch_interval=self.config["train"]["valid_epoch_interval"],
            foldername=foldername,
            propnet = propnet
        )

        directory = self.diffpo_path + "/save_model/" + current_id
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # # load model
        # model.load_state_dict(torch.load(directory + "/model_weights.pth"))

        # save model
        torch.save(model.state_dict(), directory + "/model_weights.pth")
        
        

    # predict function with bool return_po and return potential outcome if true
    def predict(self, X: np.ndarray, T0: np.ndarray = None, T1: np.ndarray = None, outcomes: np.ndarray = None) -> np.ndarray:
        """
        Predict the treatment effect using the DiffPO estimator.
        """
        # Store data for their pipeline
        data_dir = self.data_dir
        
        data, mask = self.reshape_data(X, T0, outcomes)
        
        data.to_csv(data_dir+"acic2018_norm_data/data_pp_test.csv", index=False)
        mask.to_csv(data_dir+"acic2018_mask/data_pp_test.csv", index=False)

        # Remove old files
        if os.path.exists(data_dir+"missing_ratio-0.2_seed-1_current_id-data_max-min_norm.pk"):
            os.remove(data_dir+"missing_ratio-0.2_seed-1_current_id-data_max-min_norm.pk")
        if os.path.exists(data_dir+"missing_ratio-0.2_seed-1.pk"):
            os.remove(data_dir+"missing_ratio-0.2_seed-1.pk")

        # Create folder
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # define these as variables
        nfold = 1
        current_id = "data_pp_test"
        current_id_train = "data_pp"
        seed = 1
        testmissingratio = 0.2
        nsample = 50
        perform_training = 1

        foldername = "./save/acic_fold" + str(nfold) + "_" + current_time + "/"
        # print("model folder:", foldername)
        os.makedirs(foldername, exist_ok=True)

        # Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
        training_size = 0
        _,_,test_loader = get_dataloader(
            seed=seed,
            nfold=nfold,
            batch_size=1,
            missing_ratio=testmissingratio,
            dataset_name = self.config["dataset"]["data_name"],
            current_id = current_id,
            training_size = training_size,
            data_path=data_dir,
            x_dim=X.shape[1],
        )

        # load model
        directory = self.diffpo_path + "/save_model/" + current_id_train
        os.makedirs(directory, exist_ok=True)
        model = TabCSDI(self.config, self.device).to(self.device)
        model.load_state_dict(torch.load(directory + "/model_weights.pth"))

        # get cates
        return self.evaluate(model, test_loader, nsample, foldername=foldername)
    
    def predict_outcomes(self, X: np.ndarray, T0: np.ndarray = None, T1: np.ndarray = None, outcomes: np.ndarray = None) -> np.ndarray:
        """
        Predict the potential outcomes using the DiffPO estimator.
        """
        # add outer dimension to self.pred_outcomes
        return self.pred_outcomes.cpu().numpy().reshape(self.pred_outcomes.shape[0], self.pred_outcomes.shape[1], 1)

    def explain(self, X: np.ndarray, background_samples: np.ndarray = None, explainer_limit: int = None) -> np.ndarray:
        """
        Explain the treatment effect using the EconML estimator.
        """
        if explainer_limit is None:
            explainer_limit = X.shape[0]

        return self.est.shap_values(X[:explainer_limit], background_samples=None)
    
    def infer_effect_ci(self, X, T0) -> np.ndarray:
        """
        Infer the confidence interval of the treatment effect using the EconML estimator.
        """
        cates_conf_lbs = self.cate_cis[0]
        cates_conf_ups = self.cate_cis[1]

        temp = cates_conf_lbs[T0 != 0]
        cates_conf_lbs[T0 != 0] = -cates_conf_ups[T0 != 0]
        cates_conf_ups[T0 != 0] = -temp
        return np.array([cates_conf_lbs, cates_conf_ups])
    
    def evaluate(self, model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
        # Control random seed in the current script.
        torch.manual_seed(0)
        np.random.seed(0)

        with torch.no_grad():
            model.eval()
            mse_total = 0
            mae_total = 0
            evalpoints_total = 0

            pehe_test = AverageMeter()
            y0_test = AverageMeter()
            y1_test = AverageMeter()

            # for uncertainty
            y0_samples = []
            y1_samples = []
            y0_true_list = []
            y1_true_list = []
            ite_samples = []
            ite_true_list = []
            pred_ites = []
            pred_y0s = []
            pred_y1s = []
            
            for batch_no, test_batch in enumerate(test_loader, start=1):
                # Get model outputs
                output = model.evaluate(test_batch, nsample) 
                samples, observed_data, target_mask, observed_mask, observed_tp = output

                # Extract relevant quantities
                y0_samples.append(samples[:,:,0]) 
                y1_samples.append(samples[:,:,1]) 
                ite_samples.append(samples[:,:,1] - samples[:,:,0])

                # Get point estimation through median
                est_data = torch.median(samples, dim=1).values

                # Get true ite
                obs_data = observed_data.squeeze(1)
                true_ite = obs_data[:, 2] - obs_data[:, 1] 
                ite_true_list.append(true_ite)

                # Get predicted ite
                pred_y0 = est_data[:, 0]
                pred_y1 = est_data[:, 1]
                pred_y0s.append(pred_y0)
                pred_y1s.append(pred_y1)
                y0_true_list.append(obs_data[:, 1])
                y1_true_list.append(obs_data[:, 2])
                pred_ite = pred_y1 - pred_y0
                pred_ites.append(pred_ite)

                #y0_test.update(diff_y0, obs_data.size(0))    
                #diff_y0 = np.mean((pred_y0.cpu().numpy()-obs_data[:, 1].cpu().numpy())**2)
                #y1_test.update(diff_y1, obs_data.size(0)) 
                #diff_y1 = np.mean((pred_y1.cpu().numpy()-obs_data[:, 2].cpu().numpy())**2)
                #pehe_test.update(diff_ite, obs_data.size(0))    
                #diff_ite = np.mean((true_ite.cpu().numpy()-est_ite.cpu().numpy())**2)

#---------------uncertainty estimation-------------------------
            pred_samples_y0 = torch.cat(y0_samples, dim=0)
            pred_samples_y1 = torch.cat(y1_samples, dim=0)
            pred_samples_ite = torch.cat(ite_samples, dim=0)

            truth_y0 = torch.cat(y0_true_list, dim=0) 
            truth_y1 = torch.cat(y1_true_list, dim=0) 
            truth_ite = torch.cat(ite_true_list, dim=0)

            prob_0, median_width_0 = self.compute_interval(pred_samples_y0, truth_y0)
            prob_1, median_width_1 = self.compute_interval(pred_samples_y1, truth_y1)
            prob_ite, median_width_ite = self.compute_interval(pred_samples_ite, truth_ite)

            self.cate_cis = torch.zeros(2, pred_samples_ite.shape[0], 1) # confidence intervals, dim: 2, n, dim_Y
            for i in range(pred_samples_ite.shape[0]):
                lower_quantile, upper_quantile, in_quantiles = self.check_intervel(confidence_level=0.95, y_pred= pred_samples_ite[i, :], y_true=truth_ite[i])
                self.cate_cis[0, i, 0] = lower_quantile
                self.cate_cis[1, i, 0] = upper_quantile
           
    #----------------------------------------------------------------
        pred_ites = torch.cat(pred_ites, dim=0)
        pred_y0s = torch.cat(pred_y0s, dim=0)
        pred_y1s = torch.cat(pred_y1s, dim=0)

        #np.zeros((X.shape[0], self.cfg.simulator.num_T, self.cfg.simulator.dim_Y))
        self.pred_outcomes = torch.cat([pred_y0s.unsqueeze(1), pred_y1s.unsqueeze(1)], dim=1)
        self.cate_cis = self.cate_cis.cpu().numpy()
        
        return pred_ites

    def check_intervel(self, confidence_level, y_pred, y_true):
        lower = (1 - confidence_level) / 2
        upper = 1 - lower
        lower_quantile = torch.quantile(y_pred, lower)
        upper_quantile = torch.quantile(y_pred, upper)
        in_quantiles = torch.logical_and(y_true >= lower_quantile, y_true <= upper_quantile)
        return lower_quantile, upper_quantile, in_quantiles

    def compute_interval(self, po_samples, y_true):
        counter = 0
        width_list = []
        for i in range(po_samples.shape[0]):
            lower_quantile, upper_quantile, in_quantiles = self.check_intervel(confidence_level=0.95, y_pred= po_samples[i, :], y_true=y_true[i])
            if in_quantiles == True:
                counter+=1
            width = upper_quantile - lower_quantile
            width_list.append(width.unsqueeze(0))
        prob = (counter/po_samples.shape[0])
        all_width = torch.cat(width_list, dim=0)
        median_width = torch.median(all_width, dim=0).values
        return prob, median_width