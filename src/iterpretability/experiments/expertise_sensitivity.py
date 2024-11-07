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
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import imageio
import torch
import shap

# Hydra for configuration
import hydra
from omegaconf import DictConfig, OmegaConf


class ExpertiseSensitivity(ExperimentBase):
    """
    Sensitivity analysis for varying propensity scales. This experiment will generate a .csv with the recorded metrics.
    It will also generate a gif, showing the progression on dimensionality-reduced spaces.
    """

    def __init__(
        self, cfg: DictConfig
    ) -> None:
        super().__init__(cfg)

        # Experiment specific settings
        self.alphas = cfg.alphas
        self.propensity_types = cfg.propensity_types

    def run(self) -> None:
        """
        Run the experiment.
        """
        # Log
        log.info(
            f"Starting propensity scale sensitivity experiment for dataset {self.cfg.dataset}."
        )
        
        # Main Loop
        results_data = []
   
        for seed in self.seeds:
            #for unbalancedness_exp in self.unbalancedness_exps:
            for propensity_type in self.propensity_types:
                for alpha in self.alphas:
                    log.info(
                        f"Running experiment for seed {seed} and alpha: {alpha}."
                    )

                    # Initialize the simulator
                    if self.simulation_type == "TY":
                        sim = TYSimulator(dim_X = self.X.shape[1], **self.cfg.simulator, seed=seed)
                    elif self.simulation_type == "T":
                        sim = TSimulator(dim_X = self.X.shape[1], **self.cfg.simulator, seed=seed)
                    # Overwrite propensity and nonlinearity
                    sim.alpha = alpha
                    # sim.unbalancedness_exp = unbalancedness_exp
                    sim.propensity_type = propensity_type

                    # Retrieve important features
                    self.all_important_features = sim.all_important_features
                    self.pred_features = sim.predictive_features
                    self.prog_features = sim.prognostic_features
                    self.select_features = sim.selective_features

                    # Simulate outcomes and treatment assignments
                    sim.simulate(X=self.X, outcomes=self.outcomes)
                    
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
                            
                        log.info(
                            f"Running experiment for seed {seed} and alpha: {alpha}."
                        )

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
                            alpha, 
                            "Alpha",
                            propensity_type, 
                            "Propensity Type",
                            seed,
                            split_id
                        )

        # Save results and plot
        self.save_results(metrics_df)
