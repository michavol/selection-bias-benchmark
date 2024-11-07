from src.iterpretability.experiments.experiments_base import ExperimentBase

from pathlib import Path
import os
import catenets.models as cate_models
import numpy as np
import pandas as pd
import wandb 
import random
from PIL import Image
import src.iterpretability.logger as log
from src.plotting import (
    plot_results_datasets_compare, 
    merge_pngs
)
from src.iterpretability.explain import Explainer
from src.iterpretability.datasets.data_loader import load
from src.iterpretability.simulators import (
    TYSimulator,
    TSimulator
)
from src.iterpretability.utils import (
    attribution_accuracy,
)

# For contour plotting
import umap 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import imageio
import torch
import shap

# Hydra for configuration
import hydra
from omegaconf import DictConfig, OmegaConf

class DataDimensionalitySensitivity(ExperimentBase):
    """
    Sensitivity analysis for varying number data dimensions. This experiment will generate a .csv with the recorded metrics.
    It will also compare different sample sizes.
    NOTE: Make sure that when using specific features, that the demensions work out!
    """

    def __init__(
        self, cfg: DictConfig
    ) -> None:
        super().__init__(cfg)

        # Experiment specific settings
        self.data_dims = cfg.data_dims
        self.sample_sizes = cfg.sample_sizes
        self.propensity_scales = cfg.propensity_scales
        self.sim_alpha = cfg.sim_alpha
        self.sim_propensity_type = cfg.sim_propensity_type
        self.important_feature_nums = cfg.important_feature_nums
        self.compare_axis = cfg.compare_axis


    def run(self) -> None:
        """
        Run the experiment.
        """
        # Log
        log.info(
            f"Starting cohort size sensitivity experiment for dataset {self.cfg.dataset}."
        )

        if self.compare_axis == "propensity":
            # Main Loop
            results_data = []
            for seed in self.seeds:
                for propensity_scale in self.propensity_scales:
                    for data_dim in self.data_dims:
                        # Initialize the simulator
                        random.seed(seed)
                        population = list(range(self.X.shape[1]))

                        if self.simulation_type == "TY":
                            X_curr = self.X[:,:data_dim]
                        elif self.simulation_type == "T":
                            features_to_keep = random.sample(population, data_dim)
                            X_curr = self.X[:,features_to_keep]

                        if self.simulation_type == "TY":
                            sim = TYSimulator(dim_X = X_curr.shape[1], **self.cfg.simulator, seed=seed)
                        elif self.simulation_type == "T":
                            sim = TSimulator(dim_X = X_curr.shape[1], **self.cfg.simulator, seed=seed)

                        sim.propensity_scale = propensity_scale
                        # sim.unbalancedness_exp = unbalancedness_exp
                        # sim.nonlinearity_scale = nonlinearity_scale
                        sim.propensity_type = self.sim_propensity_type
                        sim.alpha = self.sim_alpha

                        # Retrieve important features
                        self.all_important_features = sim.all_important_features
                        self.pred_features = sim.predictive_features
                        self.prog_features = sim.prognostic_features
                        self.select_features = sim.selective_features

                        # Check whether dimensions match
                        num_important_features = sim.num_important_features
                        if num_important_features > min(self.data_dims):
                            raise ValueError(
                                f"Number of important features {num_important_features} is larger than the smallest data dimension {min(self.data_dims)}."
                            )
                        
                        # Simulate outcomes and treatment assignments
                        sim.simulate(X=X_curr, outcomes=self.outcomes)
                        
                        (
                            X,
                            T,
                            Y,
                            outcomes,
                            propensities
                        ) = sim.get_simulated_data()

                        # Get splits for cross validation
                        if self.discrete_outcome:
                            Y = Y.astype(bool)
                            kf = StratifiedKFold(n_splits=self.n_splits)
                        else:
                            kf = KFold(n_splits=self.n_splits)  # Change n_splits to the number of folds you want

                        # Repeat everything for each fold
                        for split_id, (train_index, test_index) in enumerate(kf.split(X, Y)):

                            # Extract the data and split it into train and test
                            train_size = len(train_index)
                            test_size = len(test_index)
                            X_train, X_test = X[train_index], X[test_index]
                            T_train, T_test = T[train_index], T[test_index]
                            Y_train, Y_test = Y[train_index], Y[test_index]
                            outcomes_train, outcomes_test = outcomes[train_index], outcomes[test_index]
                            propensities_train, propensities_test = propensities[train_index], propensities[test_index]

                        # # Simulate outcomes and treatment assignments
                        # sim.simulate(X=X_curr, outcomes=self.outcomes)
                        
                        # (
                        #     X_full,
                        #     T_full,
                        #     Y_full,
                        #     outcomes_full,
                        #     propensities_full
                        # ) = sim.get_simulated_data()
                        
                        # # Get splits for cross validation
                        # if self.discrete_outcome:
                        #     kf = StratifiedKFold(n_splits=self.n_splits)
                        # else:
                        #     kf = KFold(n_splits=self.n_splits)  # Change n_splits to the number of folds you want

                        # # Repeat everything for each fold
                        # for split_id, (train_index, test_index) in enumerate(kf.split(X_full,Y_full)):

                        #     # Extract the data and split it into train and test
                        #     train_size_full = len(train_index)
                        #     test_size_full = len(test_index)
                        #     X_train_full, X_test_full = X_full[train_index], X_full[test_index]
                        #     T_train_full, T_test_full = T_full[train_index], T_full[test_index]
                        #     Y_train_full, Y_test_full = Y_full[train_index], Y_full[test_index]
                        #     outcomes_train_full, outcomes_test_full = outcomes_full[train_index], outcomes_full[test_index]
                        #     propensities_train_full, propensities_test_full = propensities_full[train_index], propensities_full[test_index]

                        #     log.debug(
                        #         f'Check simulated data for seed: {seed}:'
                        #         f'============================================'
                        #         f'X_train_full: {X_train_full.shape}'
                        #         f'{X_train_full}'
                        #         f'\nX_test_full: {X_test_full.shape}'
                        #         f'{X_test_full}'
                        #         f'\nT_train_full: {T_train_full.shape}'
                        #         f'{T_train_full}'
                        #         f'\nT_test_full: {T_test_full.shape}'
                        #         f'{T_test_full}'
                        #         f'\nY_train_full: {Y_train_full.shape}'
                        #         f'{Y_train_full}'
                        #         f'\nY_test_full: {Y_test_full.shape}'
                        #         f'{Y_test_full}'
                        #         f'\noutcomes_train_full: {outcomes_train_full.shape}'
                        #         f'{outcomes_train_full}'
                        #         f'\noutcomes_test_full: {outcomes_test_full.shape}'
                        #         f'{outcomes_test_full}'
                        #         f'\npropensities_train_full: {propensities_train_full.shape}'
                        #         f'{propensities_train_full}'
                        #         f'\npropensities_test_full: {propensities_test_full.shape}'
                        #         f'{propensities_test_full}'
                        #         f'\n============================================\n\n'
                        #     )

                        #     for sample_size in self.sample_sizes:
                        #         log.info(
                        #             f"Running experiment for seed {seed} and sample size {sample_size}."
                        #         )
                        #         # Define train and test sets
                        #         train_size = int(sample_size * train_size_full)
                        #         test_size = int(sample_size * test_size_full)

                        #         # Extract current training data
                        #         X_train = X_train_full[:train_size]
                        #         Y_train = Y_train_full[:train_size]
                        #         T_train = T_train_full[:train_size]
                        #         outcomes_train = outcomes_train_full[:train_size]

                        #         X_test = X_test_full[:test_size]
                        #         Y_test = Y_test_full[:test_size]
                        #         T_test = T_test_full[:test_size]
                        #         outcomes_test = outcomes_test_full[:test_size]

                            metrics_df = self.compute_metrics(
                                results_data,
                                sim,
                                X_train,
                                Y_train,
                                T_train,
                                X_test,
                                Y_test,
                                T_test,
                                outcomes_train,
                                outcomes_test,
                                propensities_train,
                                propensities_test,
                                data_dim, 
                                "Data Dimension",
                                propensity_scale, 
                                "Propensity Scale",
                                seed,
                                split_id
                            )

            # Save results and plot
            self.save_results(metrics_df, compare_axis="Propensity Scale")
        
        elif self.compare_axis == "num_features":
            # Main Loop
            results_data = []
            for seed in self.seeds:
                for important_feature_num in self.important_feature_nums:
                    for data_dim in self.data_dims:
                        # Initialize the simulator
                        random.seed(seed)
                        population = list(range(self.X.shape[1]))

                        # Overwrite the number of important features
                        # self.cfg.simulator.treatment_feature_overlap = treatment_feature_overlap
                        self.cfg.simulator.num_select_features = important_feature_num

                        if self.simulation_type == "TY":
                            self.cfg.simulator.num_pred_features = important_feature_num
                            self.cfg.simulator.num_prog_features = important_feature_num
                            X_curr = self.X[:,:data_dim]
                        elif self.simulation_type == "T":
                            features_to_keep = random.sample(population, data_dim)
                            X_curr = self.X[:,features_to_keep]

                        if self.simulation_type == "TY":
                            sim = TYSimulator(dim_X = X_curr.shape[1], **self.cfg.simulator, seed=seed)
                        elif self.simulation_type == "T":
                            sim = TSimulator(dim_X = X_curr.shape[1], **self.cfg.simulator, seed=seed)


                        # Retrieve important features
                        self.all_important_features = sim.all_important_features
                        self.pred_features = sim.predictive_features
                        self.prog_features = sim.prognostic_features
                        self.select_features = sim.selective_features

                        # Check whether dimensions match
                        num_important_features = sim.num_important_features
                        if num_important_features > min(self.data_dims):
                            raise ValueError(
                                f"Number of important features {num_important_features} is larger than the smallest data dimension {min(self.data_dims)}."
                            )
                        
                        # Simulate outcomes and treatment assignments
                        sim.simulate(X=X_curr, outcomes=self.outcomes)
                        
                        (
                            X,
                            T,
                            Y,
                            outcomes,
                            propensities
                        ) = sim.get_simulated_data()

                        # Get splits for cross validation
                        if self.discrete_outcome:
                            kf = StratifiedKFold(n_splits=self.n_splits)
                        else:
                            kf = KFold(n_splits=self.n_splits)  # Change n_splits to the number of folds you want

                        # Repeat everything for each fold
                        for split_id, (train_index, test_index) in enumerate(kf.split(X, Y)):

                            # Extract the data and split it into train and test
                            train_size = len(train_index)
                            test_size = len(test_index)
                            X_train, X_test = X[train_index], X[test_index]
                            T_train, T_test = T[train_index], T[test_index]
                            Y_train, Y_test = Y[train_index], Y[test_index]
                            outcomes_train, outcomes_test = outcomes[train_index], outcomes[test_index]
                            propensities_train, propensities_test = propensities[train_index], propensities[test_index]

                        # # Simulate outcomes and treatment assignments
                        # sim.simulate(X=X_curr, outcomes=self.outcomes)
                        
                        # (
                        #     X_full,
                        #     T_full,
                        #     Y_full,
                        #     outcomes_full,
                        #     propensities_full
                        # ) = sim.get_simulated_data()
                        
                        # # Get splits for cross validation
                        # if self.discrete_outcome:
                        #     kf = StratifiedKFold(n_splits=self.n_splits)
                        # else:
                        #     kf = KFold(n_splits=self.n_splits)  # Change n_splits to the number of folds you want

                        # # Repeat everything for each fold
                        # for split_id, (train_index, test_index) in enumerate(kf.split(X_full,Y_full)):

                        #     # Extract the data and split it into train and test
                        #     train_size_full = len(train_index)
                        #     test_size_full = len(test_index)
                        #     X_train_full, X_test_full = X_full[train_index], X_full[test_index]
                        #     T_train_full, T_test_full = T_full[train_index], T_full[test_index]
                        #     Y_train_full, Y_test_full = Y_full[train_index], Y_full[test_index]
                        #     outcomes_train_full, outcomes_test_full = outcomes_full[train_index], outcomes_full[test_index]
                        #     propensities_train_full, propensities_test_full = propensities_full[train_index], propensities_full[test_index]

                        #     log.debug(
                        #         f'Check simulated data for seed: {seed}:'
                        #         f'============================================'
                        #         f'X_train_full: {X_train_full.shape}'
                        #         f'{X_train_full}'
                        #         f'\nX_test_full: {X_test_full.shape}'
                        #         f'{X_test_full}'
                        #         f'\nT_train_full: {T_train_full.shape}'
                        #         f'{T_train_full}'
                        #         f'\nT_test_full: {T_test_full.shape}'
                        #         f'{T_test_full}'
                        #         f'\nY_train_full: {Y_train_full.shape}'
                        #         f'{Y_train_full}'
                        #         f'\nY_test_full: {Y_test_full.shape}'
                        #         f'{Y_test_full}'
                        #         f'\noutcomes_train_full: {outcomes_train_full.shape}'
                        #         f'{outcomes_train_full}'
                        #         f'\noutcomes_test_full: {outcomes_test_full.shape}'
                        #         f'{outcomes_test_full}'
                        #         f'\npropensities_train_full: {propensities_train_full.shape}'
                        #         f'{propensities_train_full}'
                        #         f'\npropensities_test_full: {propensities_test_full.shape}'
                        #         f'{propensities_test_full}'
                        #         f'\n============================================\n\n'
                        #     )

                        #     for sample_size in self.sample_sizes:
                        #         log.info(
                        #             f"Running experiment for seed {seed} and sample size {sample_size}."
                        #         )
                        #         # Define train and test sets
                        #         train_size = int(sample_size * train_size_full)
                        #         test_size = int(sample_size * test_size_full)

                        #         # Extract current training data
                        #         X_train = X_train_full[:train_size]
                        #         Y_train = Y_train_full[:train_size]
                        #         T_train = T_train_full[:train_size]
                        #         outcomes_train = outcomes_train_full[:train_size]

                        #         X_test = X_test_full[:test_size]
                        #         Y_test = Y_test_full[:test_size]
                        #         T_test = T_test_full[:test_size]
                        #         outcomes_test = outcomes_test_full[:test_size]

                            metrics_df = self.compute_metrics(
                                results_data,
                                sim,
                                X_train,
                                Y_train,
                                T_train,
                                X_test,
                                Y_test,
                                T_test,
                                outcomes_train,
                                outcomes_test,
                                propensities_train,
                                propensities_test,
                                data_dim, 
                                "Data Dimension",
                                important_feature_num, 
                                "Num Important Features",
                                seed,
                                split_id
                            )

            # Save results and plot
            self.save_results(metrics_df, compare_axis="Num Important Features")

        else:
            raise ValueError(
                f"Invalid compare_axis: {self.compare_axis}."
            )