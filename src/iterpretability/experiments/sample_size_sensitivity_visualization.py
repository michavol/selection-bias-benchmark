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
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import imageio
import torch
import shap

# Hydra for configuration
import hydra
from omegaconf import DictConfig, OmegaConf
class SampleSizeSensitivityVisualization(ExperimentBase):
    """
    Sensitivity analysis for varying numbers of samples. This experiment will generate a .csv with the recorded metrics.
    It will also generate a gif, showing the progression on dimensionality-reduced spaces.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        # Experiment specific settings
        self.sample_sizes = cfg.sample_sizes
        self.propensity_scales = cfg.propensity_scales

        # Create directory and names for output
        self.ind_imgs_dir = self.results_path / Path("individual_images")
        self.ind_imgs_dir.mkdir(parents=True, exist_ok=True)


    def run(self) -> None:
        # Log setting
        log.info(
            f"Using dataset {self.cfg.dataset}."
        )

        # Initialize the simulator
        seed = self.cfg.seeds[0]
        if self.simulation_type == "TY":
            sim = TYSimulator(dim_X = self.X.shape[1], **self.cfg.simulator, seed = seed)
        elif self.simulation_type == "T":
            sim = TSimulator(dim_X = self.X.shape[1], **self.cfg.simulator, seed = seed)

        # Retrieve important features
        all_important_features = sim.all_important_features

        for k, propensity_scale in enumerate(self.propensity_scales):
            # Simulate outcomes and treatment assignments
            sim.propensity_scale = propensity_scale
            sim.simulate(X=self.X, outcomes=self.outcomes)

            # Extract the data and split it into train and test
            (
                X_train_full, X_test_full, 
                T_train_full, T_test_full, 
                Y_train_full, Y_test_full, 
                outcomes_train_full, outcomes_test_full, 
                propensities_train_full, propensities_test_full
            ) = sim.get_simulated_data(train_ratio=self.cfg.train_ratio)
            train_full_size = X_train_full.shape[0]
            test_full_size = X_test_full.shape[0]

            # Extract potential outcomes (pos) and propensities for special setting with num_T=2
            po0_train_full = outcomes_train_full[:,0,0].squeeze()
            po0_test_full = outcomes_test_full[:,0,0].squeeze()
            po1_train_full = outcomes_train_full[:,1,0].squeeze()
            po1_test_full = outcomes_test_full[:,1,0].squeeze()
            propensity_train_full = propensities_train_full[:,1].squeeze()

            log.debug(
                f'Check simulated data for seed: {seed}:'
                f'============================================'
                f'X_train_full: {X_train_full.shape}'
                # f'{X_train}'
                f'\X_test_full: {X_test_full.shape}'
                # f'{X_test}'
                f'\T_train_full: {T_train_full.shape}'
                # f'{T_train}'
                f'\T_test_full: {T_test_full.shape}'
                # f'{T_test}'
                f'\Y_train_full: {Y_train_full.shape}'
                # f'{Y_train}'
                f'\Y_test_full: {Y_test_full.shape}'
                # f'{Y_test}'
                f'\po0_train_full: {po0_train_full.shape}'
                # f'{outcomes_train}'
                f'\po0_test_full: {po0_test_full.shape}'
                # f'{outcomes_test}'
                f'\po1_test_full: {po1_test_full.shape}'
                # f'{propensities_train}'
                f'\propensity_train_full: {propensity_train_full.shape}'
                # f'{propensities_test}'
                f'\n============================================\n\n'
            )

            # Use full training set with added noisy samples as focused samples of space
            background_samples = np.concatenate([X_train_full, X_test_full], axis=0)

            # Reduce samples to two dimensions for plotting using umap
            if self.cfg.dim_reduction_method == "umap":
                reducer = umap.UMAP(min_dist=2, n_neighbors=30, spread=3)
                reducer_shap = umap.UMAP(min_dist=4, n_neighbors=30, spread=5)

            elif self.cfg.dim_reduction_method == "pca":
                reducer = PCA(n_components=2)
                reducer_shap = PCA(n_components=2)

            elif self.cfg.dim_reduction_method == "tsne":
                raise Exception("t-SNE not supported for this analysis. Does not offer .transform() method.")

            else:
                raise Exception("Unknown dimensionality reduction method.")

            # Fit on background samples and training data
            if self.cfg.dim_reduction_on_important_features:
                reducer = reducer.fit(background_samples[:, all_important_features])
                background_samples_2d = reducer.transform(background_samples[:, all_important_features])
            else:
                reducer = reducer.fit(background_samples)
                background_samples_2d = reducer.transform(background_samples)

            train_samples_2d = background_samples_2d[:train_full_size]
            test_samples_2d = background_samples_2d[train_full_size:] 

            # Get learners
            self.learners = self.get_learners(num_features=self.X.shape[1], seed=seed)

            # Train learners
            self.train_learners(X_train_full, Y_train_full, T_train_full)

            # Get learner explanations
            learner_explanations_background = self.get_learner_explanations(X=background_samples, 
                                                                            return_explainer_names=False, 
                                                                            ignore_explainer_limit=True)
        
            # Get shap for background samples
            shap_values_background = learner_explanations_background[self.cfg.model_names[0]]
            
            # Extract top k columns in terms of mean absolute shap values
            background_top_important_features = np.argsort(np.abs(shap_values_background).mean(0))[::-1][:self.cfg.top_k_shap_features]
        
            # Perform logistic regression for propensity
            est_prop = LogisticRegression().fit(X_train_full, T_train_full)
            explainer_prop = shap.LinearExplainer(est_prop, X_train_full)
            shap_values_prop_background = explainer_prop.shap_values(background_samples)

            # Extract top k columns in terms of mean absolute shap values
            prop_background_top_important_features = np.argsort(np.abs(shap_values_prop_background).mean(0))[::-1][:self.cfg.top_k_shap_features]
            
            # Combine the important shap features to get a shared space for the background samples
            common_shap_values_background = np.concatenate([shap_values_background[:, background_top_important_features], shap_values_prop_background[:, prop_background_top_important_features]], axis=1)

            # Fit umap reducer for shap background samples 
            if self.cfg.dim_reduction_on_important_features:
                reducer = reducer_shap.fit(common_shap_values_background[:, all_important_features])
                common_shap_values_background_2d = reducer_shap.transform(common_shap_values_background[:, all_important_features])

            else:
                reducer = reducer_shap.fit(common_shap_values_background)
                common_shap_values_background_2d = reducer_shap.transform(common_shap_values_background)

            # Split the shap values into train and test
            common_shap_values_train_2d = common_shap_values_background_2d[:train_full_size]
            common_shap_values_test_2d = common_shap_values_background_2d[train_full_size:]

            # Get formatter for axis ticks
            formatter = FuncFormatter(lambda x, pos: f'{x:.1f}')

            # Get scores for normalization
            # Find min and max values for propensities and cate predictions for plotting
            num_levels = self.cfg.num_levels
            vmin_prop = 0
            vmax_prop = 1
            eff_train_full = (outcomes_train_full[:,1,0] - outcomes_train_full[:,0,0]).squeeze()
            eff_test_full = (outcomes_test_full[:,1,0] - outcomes_test_full[:,0,0]).squeeze()
            vmin_cate = min(eff_train_full.min(), eff_test_full.min())
            vmax_cate = max(eff_train_full.max(), eff_test_full.max())
            vmin_diff = 0
            vmax_diff = vmax_cate-vmin_cate
            norm_prop = Normalize(vmin=vmin_prop, vmax=vmax_prop)
            norm_cate = Normalize(vmin=vmin_cate, vmax=vmax_cate)
            norm_diff = Normalize(vmin=vmin_diff, vmax=vmax_diff)
            levels_prop = np.linspace(vmin_prop, vmax_prop, num_levels, endpoint=True)
            levels_cate = np.linspace(vmin_cate, vmax_cate, num_levels, endpoint=True)
            levels_diff = np.linspace(vmin_diff, vmax_diff, num_levels, endpoint=True)

            # Iterate over the sample sizes
            cohort_size_full = X_train_full.shape[0]
            for m, cohort_size_perc in enumerate(self.sample_sizes):
                # Get the current cohort size
                cohort_size = int(cohort_size_perc * cohort_size_full)
                cohort_size_train = int(self.cfg.train_ratio * cohort_size)
                cohort_size_test = cohort_size - cohort_size_train

                # Get a subset of the training data
                X_train = X_train_full[:cohort_size_train]
                T_train = T_train_full[:cohort_size_train]
                Y_train = Y_train_full[:cohort_size_train]

                # Get subsets for test data
                X_test = X_test_full[:cohort_size_test]
                T_test = T_test_full[:cohort_size_test]
                Y_test = Y_test_full[:cohort_size_test]

                log.info(f"Now working with a cohort size of {cohort_size}/{X_train_full.shape[0]}...")
                log.info("Fitting and explaining learners...")

                # Train learners
                self.train_learners(X_train, Y_train, T_train)

                # Make logistic regression for propensity
                est_prop = LogisticRegression().fit(X_train, T_train)
                explainer_prop = shap.LinearExplainer(est_prop, X_train)

                # Get true effect and propensity
                sim.simulate(X=background_samples, outcomes=self.outcomes)

                # Extract the data and split it into train and test
                (
                    _, 
                    _, 
                    _, 
                    outcomes_background,
                    propensities_background
                ) = sim.get_simulated_data(train_ratio=1)

                # Extract true effect and propensity
                eff_background = (outcomes_background[:,1,0] - outcomes_background[:,0,0]).squeeze()
                prop_background = (propensities_background[:,1]).squeeze()

                # Get cate estimator predictions
                est_eff = self.learners[self.cfg.model_names[0]]
                cate_pred_test = est_eff.predict(X=X_test)
                cate_pred_train = est_eff.predict(X=X_train)

                # Get predictions on background samples for the current trained model
                p_eff_background = est_eff.predict(X=background_samples).squeeze()
                p_prop_background = est_prop.predict_proba(background_samples)[:, 1]

                # Get absolute differences of cate prediction for the background samples
                cate_diff = np.abs(p_eff_background - eff_background)
                # prop_diff = np.abs(p_prop_background - prop_background)

                print(p_eff_background.shape, eff_background.shape, cate_diff.shape)
                outcomes = [p_prop_background, prop_background, p_eff_background, eff_background, cate_diff]
                titles = ['prop', 'prop_true', 'cate', 'true cate', 'abs cate diff']

                # Set up plots
                cbar_ratio = 0.03
                fig = plt.figure(1, figsize=(26,5))
                gs = gridspec.GridSpec(1, 8, width_ratios=[1, 1, cbar_ratio, 1, 1, cbar_ratio, 1, cbar_ratio], height_ratios=[1], wspace=0.18, hspace=1)
                axs = [plt.subplot(gs[0, i]) for i in [0,1,3,4,6]]

                fig_shap = plt.figure(2, figsize=(26,5))
                gs_shap = gridspec.GridSpec(1, 8, width_ratios=[1, 1, cbar_ratio, 1, 1, cbar_ratio, 1, cbar_ratio], height_ratios=[1], wspace=0.18, hspace=1)
                axs_shap = [plt.subplot(gs_shap[0, i]) for i in [0,1,3,4,6]]

                
                # If there are any non-finite elements in the outcomes, remove them from the background_samples and the other outcomes and raise a warning
                for j, outcome in enumerate(outcomes):
                    if type(outcome) == torch.Tensor:
                        outcome = outcome.cpu().detach().numpy()

                    # outcome = np.array(outcome)
                    # if not np.all(np.isfinite(outcome)):
                    #     print("-----------")
                    #     print(j, np.sum(~np.isfinite(outcome)))
                    #     log.warning(f'Found non-finite elements in outcomes. Removing {np.sum(~np.isfinite(outcome))} elements.')
                    #     mask = np.isfinite(outcome)
                    #     background_samples_2d = background_samples_2d[mask]
                    #     for j in range(len(outcomes)):
                    #         outcomes[j] = outcomes[j][mask]
                    #     break

                    # Plot settings
                    cmap_contour = "viridis" #"viridis"
                    cmap_contour_prop = "plasma"
                    cmap_scatter = "viridis"
                    cmap_scatter_prop = "plasma"
                    cmap_diff = "coolwarm"
                    marker = 'o'
                    marker_prop = 'o'

                    s = 17 # 4 for many samples
                    alpha = 1
                    edgecolors = 'black'
                    linewidths = 0.8
                    
                    if j <= 1:
                        plt.figure(1)
                        # tcf_prop = axs[j].scatter(background_samples_2d[:,0], 
                        #                 background_samples_2d[:,1], 
                        #                 c=outcome.ravel(), 
                        #                 cmap=cmap_contour_prop, 
                        #                 s=s+100, 
                        #                 alpha=0.2,
                        #                 norm=norm_prop,
                        #                 edgecolors=None,
                        #                 marker="o", 
                        #                 linewidths=linewidths)
                        
                        tcf_prop = axs[j].tricontourf(background_samples_2d[:,0], 
                                                    background_samples_2d[:,1], 
                                                    outcome.ravel(), 
                                                    norm=norm_prop,
                                                    levels=levels_prop,
                                                    extend='both',
                                                    alpha=alpha,
                                                    cmap=cmap_contour_prop)
                        
                        cbar_ax_prop = plt.subplot(gs[0, 2])
                        fig.colorbar(tcf_prop, cax=cbar_ax_prop, extend = "both")
                        # cbar_ax_prop.yaxis.tick_left()
                        cbar_ax_prop.yaxis.set_major_formatter(formatter)

                        plt.figure(2)
                        # tcf_shap_prop = axs_shap[j].scatter(common_shap_values_background_2d[:,0], 
                        #                 common_shap_values_background_2d[:,1], 
                        #                 c=outcome.ravel(), 
                        #                 cmap=cmap_contour_prop, 
                        #                 s=s+100, 
                        #                 alpha=0.2,
                        #                 norm=norm_prop,
                        #                 edgecolors=None,
                        #                 marker="o", 
                        #                 linewidths=linewidths)
                        
                        tcf_shap_prop = axs_shap[j].tricontourf(common_shap_values_background_2d[:,0], 
                                                        common_shap_values_background_2d[:,1], 
                                                        outcome.ravel(), 
                                                        norm=norm_prop, 
                                                        levels=levels_prop,
                                                        extend='both',
                                                        alpha=alpha,
                                                        cmap=cmap_contour_prop)
                        
                        cbar_ax_prop_shap = plt.subplot(gs_shap[0, 2])
                        fig_shap.colorbar(tcf_shap_prop, cax=cbar_ax_prop_shap, extend = "both")
                        # cbar_ax_prop_shap.yaxis.tick_left()
                        cbar_ax_prop_shap.yaxis.set_major_formatter(formatter)


                    elif j > 1 and j <= 3:
                        plt.figure(1)
                        tcf_cate = axs[j].tricontourf(background_samples_2d[:,0], 
                                                    background_samples_2d[:,1], 
                                                    outcome.ravel(), 
                                                    norm=norm_cate,
                                                    levels=levels_cate,
                                                    extend='both',
                                                    alpha=alpha,
                                                    cmap=cmap_contour)

                        cbar_ax_cate = plt.subplot(gs[0, 5])
                        fig.colorbar(tcf_cate, cax=cbar_ax_cate, extend = "both")
                        cbar_ax_cate.yaxis.set_major_formatter(formatter)
                        
                        plt.figure(2)
                        tcf_shap_cate = axs_shap[j].tricontourf(common_shap_values_background_2d[:,0], 
                                                        common_shap_values_background_2d[:,1], 
                                                        outcome.ravel(), 
                                                        norm=norm_cate,
                                                        levels=levels_cate,
                                                        extend='both',
                                                        alpha=alpha,
                                                        cmap=cmap_contour)
                        
                        cbar_ax_cate_shap = plt.subplot(gs_shap[0, 5])
                        fig_shap.colorbar(tcf_shap_cate, cax=cbar_ax_cate_shap, extend = "both")
                        cbar_ax_cate_shap.yaxis.set_major_formatter(formatter)
                        

                    elif j == 4:
                        plt.figure(1)
                        tcf_diff = axs[j].tricontourf(background_samples_2d[:,0],
                                                    background_samples_2d[:,1],
                                                    outcome.ravel(),
                                                    norm=norm_diff,
                                                    levels=levels_diff,
                                                    extend='both',
                                                    alpha=alpha,
                                                    cmap=cmap_diff)
                        
                        cbar_ax_diff = plt.subplot(gs[0, 7])
                        fig.colorbar(tcf_diff, cax=cbar_ax_diff, extend = "both")
                        cbar_ax_diff.yaxis.set_major_formatter(formatter)
                        
                        plt.figure(2)
                        tcf_shap_diff = axs_shap[j].tricontourf(common_shap_values_background_2d[:,0],
                                                        common_shap_values_background_2d[:,1],
                                                        outcome.ravel(),
                                                        norm=norm_diff,
                                                        levels=levels_diff,
                                                        extend='both',
                                                        alpha=alpha,
                                                        cmap=cmap_diff)
                        
                        cbar_ax_diff_shap = plt.subplot(gs_shap[0, 7])
                        fig_shap.colorbar(tcf_shap_diff, cax=cbar_ax_diff_shap, extend = "both")
                        cbar_ax_diff_shap.yaxis.set_major_formatter(formatter)

                    if j == 0:
                        plt.figure(1)
                        axs[j].scatter(train_samples_2d[:cohort_size_train,0], train_samples_2d[:cohort_size_train,1],
                                        c=T_train, 
                                        cmap=cmap_scatter_prop, 
                                        s=s, 
                                        edgecolors=edgecolors,
                                        marker=marker_prop, 
                                        linewidths=linewidths)
                        
                        plt.figure(2)
                        axs_shap[j].scatter(common_shap_values_train_2d[:cohort_size_train,0], common_shap_values_train_2d[:cohort_size_train,1],
                                            c=T_train, 
                                            cmap=cmap_scatter_prop, 
                                            s=s, 
                                            edgecolors=edgecolors, 
                                            marker=marker_prop,
                                            linewidths=linewidths)

                    elif j == 1:
                        plt.figure(1)
                        axs[j].scatter(test_samples_2d[:cohort_size_test,0], test_samples_2d[:cohort_size_test,1], 
                                        c=T_test, 
                                        cmap=cmap_scatter_prop, 
                                        s=s, 
                                        edgecolors=edgecolors, 
                                        marker=marker_prop,
                                        linewidths=linewidths)
                        
                        plt.figure(2)
                        axs_shap[j].scatter(common_shap_values_test_2d[:cohort_size_test,0], common_shap_values_test_2d[:cohort_size_test,1], 
                                        c=T_test, 
                                        cmap=cmap_scatter_prop, 
                                        s=s, 
                                        edgecolors=edgecolors, 
                                        marker=marker_prop,
                                        linewidths=linewidths)

                    elif j == 2:
                        plt.figure(1)
                        axs[j].scatter(train_samples_2d[:cohort_size_train,0], train_samples_2d[:cohort_size_train,1],
                                        c=cate_pred_train, 
                                        cmap=cmap_scatter, 
                                        s=s, 
                                        edgecolors=edgecolors, 
                                        marker=marker,
                                        linewidths=linewidths)
                        
                        plt.figure(2)
                        axs_shap[j].scatter(common_shap_values_train_2d[:cohort_size_train,0], common_shap_values_train_2d[:cohort_size_train,1],
                                            c=cate_pred_train, 
                                            cmap=cmap_scatter, 
                                            s=s, 
                                            edgecolors=edgecolors,
                                            marker=marker, 
                                            linewidths=linewidths)

                    elif j == 3:
                        plt.figure(1)
                        axs[j].scatter(test_samples_2d[:cohort_size_test,0], test_samples_2d[:cohort_size_test,1], 
                                        c=cate_pred_test, 
                                        cmap=cmap_scatter, 
                                        s=s, 
                                        edgecolors=edgecolors, 
                                        marker=marker,
                                        linewidths=linewidths)
                        
                        plt.figure(2)
                        axs_shap[j].scatter(common_shap_values_test_2d[:cohort_size_test,0], common_shap_values_test_2d[:cohort_size_test,1], 
                                        c=cate_pred_test, 
                                        cmap=cmap_scatter, 
                                        s=s, 
                                        edgecolors=edgecolors, 
                                        marker=marker,
                                        linewidths=linewidths)
                    
                    # Set titles and limits for the plots
                    plt.figure(1)
                    axs[j].set_xlim([background_samples_2d[:,0].min(), background_samples_2d[:,0].max()])
                    axs[j].set_ylim([background_samples_2d[:,1].min(), background_samples_2d[:,1].max()])
                
                    if k == 0:
                        if j == 0:
                            axs[j].set_title(titles[j]+f", pps = {propensity_scale}")
                        else:
                            axs[j].set_title(titles[j])
                    else:
                        axs[0].set_title(f"pps = {propensity_scale}")

                    plt.figure(2)
                    axs_shap[j].set_xlim([common_shap_values_background_2d[:,0].min(), common_shap_values_background_2d[:,0].max()])
                    axs_shap[j].set_ylim([common_shap_values_background_2d[:,1].min(), common_shap_values_background_2d[:,1].max()])

                    if k == 0:
                        if j == 0:
                            axs_shap[j].set_title(titles[j]+f", pps = {propensity_scale}")
                        else:
                            axs_shap[j].set_title(titles[j])
                    else:
                        axs_shap[0].set_title(f"pps = {propensity_scale}")


                plt.figure(1)
                fig.suptitle('')
                if k == 0:
                    fig.suptitle(f"Sample Size Sensitivity, N = {cohort_size}", fontsize=18, y=1.002)

                plt.figure(2)
                fig_shap.suptitle('')
                if k == 0:
                    fig_shap.suptitle(f"Sample Size Sensitivity, N = {cohort_size}", fontsize=18, y=1.002)

                # Turn off ticks and spines for plots in between colorbars
                plt.figure(1)
                for ax in axs:
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                plt.figure(2)
                for ax in axs_shap:
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                plt.figure(1)
                # fig.tight_layout()
                fig.savefig(self.ind_imgs_dir / f'{self.cfg.model_names[0]}_pps_{propensity_scale}_cs_{cohort_size_perc}_raw.png')

                plt.figure(2)
                # fig_shap.tight_layout()
                fig_shap.savefig(self.ind_imgs_dir / f'{self.cfg.model_names[0]}_pps_{propensity_scale}_cs_{cohort_size_perc}_shap.png')

        
        # Gifs
        # Load the images in nested list, outer list is for each cohort size, inner list is for each propensity scale
        images_raw = []
        images_shap = []
        for m, cohort_size_perc in enumerate(self.sample_sizes):
            images_raw.append([])
            images_shap.append([])
            for k, propensity_scale in enumerate(self.propensity_scales):
                images_raw[m].append(Image.open(self.ind_imgs_dir / f'{self.cfg.model_names[0]}_pps_{propensity_scale}_cs_{cohort_size_perc}_raw.png'))
                images_shap[m].append(Image.open(self.ind_imgs_dir / f'{self.cfg.model_names[0]}_pps_{propensity_scale}_cs_{cohort_size_perc}_shap.png'))

        # Merge across the propensity scales using the merge_pngs function
        images_raw_merged = []
        images_shap_merged = []
        for m, cohort_size_perc in enumerate(self.sample_sizes):
            images_raw_merged.append(merge_pngs(images_raw[m], axis="vertical"))
            images_shap_merged.append(merge_pngs(images_shap[m], axis="vertical"))

        # Save the merged images as gifs
        imageio.mimsave(self.results_path / f'visualization_raw_{self.cfg.model_names[0]}.gif', images_raw_merged, fps=0.2)
        imageio.mimsave(self.results_path / f'visualization_shap_{self.cfg.model_names[0]}.gif', images_shap_merged, fps=0.2)

        