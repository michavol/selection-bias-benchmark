from src.iterpretability.experiments.experiments_base import ExperimentBase

from pathlib import Path
import os
import catenets.models as cate_models
import numpy as np
import pandas as pd
import wandb 
from PIL import Image
import src.iterpretability.logger as log
from src.plotting import (
    plot_results_datasets_compare, 
    merge_pngs
)
from src.iterpretability.explain import Explainer
from src.iterpretability.datasets.data_loader import load
from src.iterpretability.simulators import (
    SimulatorBase,
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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
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

class SampleSizeSensitivity(ExperimentBase):
    """
    Sensitivity analysis for varying numbers of samples. This experiment will generate a .csv with the recorded metrics.
    It will also generate a gif, showing the progression on dimensionality-reduced spaces.
    """

    def __init__(
        self, cfg: DictConfig
    ) -> None:
        super().__init__(cfg)

        # Experiment specific settings
        self.sample_sizes = cfg.sample_sizes
        self.unbalancedness_exps = cfg.unbalancedness_exps
        self.propensity_scales = cfg.propensity_scales
        self.sim_alpha = cfg.sim_alpha
        self.sim_propensity_type = cfg.sim_propensity_type
        
    def run(self) -> None:
        """
        Run the experiment.
        """
        # Log
        log.info(
            f"Starting cohort size sensitivity experiment for dataset {self.cfg.dataset}."
        )

        # Perform the experiment
        results_data = []
        for seed in self.seeds:
            # try:
            for propensity_scale in self.propensity_scales:
                
                # Initialize the simulator
                if self.simulation_type == "TY":
                    sim = TYSimulator(dim_X = self.X.shape[1], **self.cfg.simulator, seed=seed)
                elif self.simulation_type == "T":
                    sim = TSimulator(dim_X = self.X.shape[1], **self.cfg.simulator, seed=seed)

                # Update unbalancedness
                # sim.unbalancedness_exp = unbalancedness_exp

                sim.propensity_scale = propensity_scale
                # sim.unbalancedness_exp = unbalancedness_exp
                # sim.nonlinearity_scale = nonlinearity_scale
                sim.propensity_type = self.sim_propensity_type
                sim.alpha = self.sim_alpha

                # Simulate outcomes and treatment assignments
                sim.simulate(X=self.X, outcomes=self.outcomes)
                
                (
                    X_full,
                    T_full,
                    Y_full,
                    outcomes_full,
                    propensities_full
                ) = sim.get_simulated_data()

                # Retrieve important features
                self.all_important_features = sim.all_important_features
                self.pred_features = sim.predictive_features
                self.prog_features = sim.prognostic_features
                self.select_features = sim.selective_features


                # Get splits for cross validation
                if self.discrete_outcome:
                    Y = Y.astype(bool)
                    kf = StratifiedKFold(n_splits=self.n_splits)
                else:
                    kf = KFold(n_splits=self.n_splits)  # Change n_splits to the number of folds you want

                # Repeat everything for each fold
                for split_id, (train_index, test_index) in enumerate(kf.split(X_full, Y_full)):

                    # Extract the data and split it into train and test
                    train_size_full = len(train_index)
                    test_size_full = len(test_index)
                    X_train_full, X_test_full = X_full[train_index], X_full[test_index]
                    T_train_full, T_test_full = T_full[train_index], T_full[test_index]
                    Y_train_full, Y_test_full = Y_full[train_index], Y_full[test_index]
                    outcomes_train_full, outcomes_test_full = outcomes_full[train_index], outcomes_full[test_index]
                    propensities_train_full, propensities_test_full = propensities_full[train_index], propensities_full[test_index]

                    log.debug(
                        f'Check simulated data for seed: {seed}:'
                        f'============================================'
                        f'X_train_full: {X_train_full.shape}'
                        f'{X_train_full}'
                        f'\nX_test_full: {X_test_full.shape}'
                        f'{X_test_full}'
                        f'\nT_train_full: {T_train_full.shape}'
                        f'{T_train_full}'
                        f'\nT_test_full: {T_test_full.shape}'
                        f'{T_test_full}'
                        f'\nY_train_full: {Y_train_full.shape}'
                        f'{Y_train_full}'
                        f'\nY_test_full: {Y_test_full.shape}'
                        f'{Y_test_full}'
                        f'\noutcomes_train_full: {outcomes_train_full.shape}'
                        f'{outcomes_train_full}'
                        f'\noutcomes_test_full: {outcomes_test_full.shape}'
                        f'{outcomes_test_full}'
                        f'\npropensities_train_full: {propensities_train_full.shape}'
                        f'{propensities_train_full}'
                        f'\npropensities_test_full: {propensities_test_full.shape}'
                        f'{propensities_test_full}'
                        f'\n============================================\n\n'
                    )

                    for sample_size in self.sample_sizes:
                        log.info(
                            f"Running experiment for seed {seed} and sample size {sample_size}."
                        )
                        # Define train and test sets
                        train_size = int(sample_size * train_size_full)
                        test_size = int(sample_size * test_size_full)

                        # Extract current training data
                        X_train = X_train_full[:train_size]
                        Y_train = Y_train_full[:train_size]
                        T_train = T_train_full[:train_size]
                        outcomes_train = outcomes_train_full[:train_size]
                        propensities_train = propensities_train_full[:train_size]

                        X_test = X_test_full[:test_size]
                        Y_test = Y_test_full[:test_size]
                        T_test = T_test_full[:test_size]
                        outcomes_test = outcomes_test_full[:test_size]
                        propensities_test = propensities_test_full[:test_size]

                        try:
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
                                sample_size, 
                                "Sample Portion",
                                propensity_scale, 
                                "Propensity Scale",
                                seed,
                                split_id
                            )
                        except Exception as e:
                            # Code to run for any other exception
                            print("One Sample size had to be skipped")
                            print("Error:", e)

        # Save results and plot
        self.save_results(metrics_df)

