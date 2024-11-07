import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd 
import numpy as np
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from PIL import Image
import sys 

# Hydra for configuration
import hydra
from omegaconf import DictConfig, OmegaConf
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter

# Custom formatter function
def custom_formatter(x, pos):
    if x.is_integer():
        return f'{int(x)}'
    elif x==0.5:
        return r'$1/2$'
    elif x==0.25:
        return r'$1/4$'
    # Do a diagonal fraction instead

    else:
        return f'{x:.2f}'

cblind_palete = sns.color_palette("colorblind", as_cmap=True)
learner_colors = {
    "Torch_SLearner": cblind_palete[0],
    "Torch_TLearner": cblind_palete[1],
    "Torch_XLearner": cblind_palete[2],
    "Torch_TARNet": cblind_palete[3],
    'Torch_CFRNet_0.01': cblind_palete[4],
    "Torch_CFRNet_0.001": cblind_palete[6],
    'Torch_CFRNet_0.0001': cblind_palete[9],
    'Torch_ActionNet': cblind_palete[7],
    "Torch_DRLearner": cblind_palete[8],
    "Torch_RALearner": cblind_palete[9],
    "Torch_DragonNet": cblind_palete[5],
    "Torch_DragonNet_2": cblind_palete[5],
    "Torch_DragonNet_4": cblind_palete[3],
    "Torch_ULearner": cblind_palete[6],
    "Torch_PWLearner": cblind_palete[7],
    "Torch_RLearner": cblind_palete[8],
    "Torch_FlexTENet": cblind_palete[9],
    "EconML_CausalForestDML": cblind_palete[2],
    "EconML_DML": cblind_palete[0],
    "EconML_DMLOrthoForest": cblind_palete[1],
    "EconML_DRLearner": cblind_palete[6],
    "EconML_DROrthoForest": cblind_palete[9],
    "EconML_ForestDRLearner": cblind_palete[7],
    "EconML_LinearDML": cblind_palete[8],
    "EconML_LinearDRLearner": cblind_palete[5],
    "EconML_SparseLinearDML": cblind_palete[3],
    "EconML_SparseLinearDRLearner": cblind_palete[4],
    "EconML_XLearner_Lasso": cblind_palete[7],
    "EconML_TLearner_Lasso": cblind_palete[8],
    "EconML_SLearner_Lasso": cblind_palete[9],
    "DiffPOLearner": cblind_palete[0],
    "Truth": cblind_palete[9],
}

learner_linestyles = {
    "Torch_SLearner": "-",
    "Torch_TLearner": "--",
    "Torch_XLearner": ":",
    "Torch_TARNet": "-.",
    "Torch_DragonNet": "--",
    "Torch_DragonNet_2": "-",
    "Torch_DragonNet_4": "-.",
    "Torch_XLearner": "--",
    "Torch_CFRNet_0.01": "-",
    "Torch_CFRNet_0.001": ":",
    "Torch_CFRNet_0.0001": "--",
    "Torch_DRLearner": "-",
    "Torch_RALearner": "--",
    "Torch_ULearner": "-",
    "Torch_PWLearner": "-",
    "Torch_RLearner": "-",
    "Torch_FlexTENet": "-",
    'Torch_ActionNet': "-",
    "EconML_CausalForestDML": "-",
    "EconML_DML": "--",
    "EconML_DMLOrthoForest": ":",
    "EconML_DRLearner": "-.",
    "EconML_DROrthoForest": "--",
    "EconML_ForestDRLearner": "-.",
    "EconML_LinearDML": ":",
    "EconML_LinearDRLearner": "-",
    "EconML_SparseLinearDML": "--",
    "EconML_SparseLinearDRLearner": ":",
    "EconML_SLearner_Lasso": "-.",
    "EconML_TLearner_Lasso": "--",
    "EconML_XLearner_Lasso": "-",
    "DiffPOLearner": "-.",
    "Truth": ":",
}


learner_markers = {
    "Torch_SLearner": "d",
    "Torch_TLearner": "o",
    "Torch_XLearner": "^",
    "Torch_TARNet": "*",
    "Torch_DragonNet": "x",
    "Torch_DragonNet_2": "o",
    "Torch_DragonNet_4": "*",
    "Torch_XLearner": "D",
    "Torch_CFRNet_0.01": "8",
    "Torch_CFRNet_0.001": "s",
    "Torch_CFRNet_0.0001": "x",
    "Torch_DRLearner": "x",
    "Torch_RALearner": "H",
    "Torch_ULearner": "x",
    "Torch_PWLearner": "*",
    "Torch_RLearner": "*",
    "Torch_FlexTENet": "*",
    'Torch_ActionNet': "*",
    "EconML_CausalForestDML": "d",
    "EconML_DML": "o",
    "EconML_DMLOrthoForest": "^",
    "EconML_DRLearner": "*",
    "EconML_DROrthoForest": "D",
    "EconML_ForestDRLearner": "8",
    "EconML_LinearDML": "s",
    "EconML_LinearDRLearner": "x",
    "EconML_SparseLinearDML": "x",
    "EconML_SparseLinearDRLearner": "H",
    "EconML_TLearner_Lasso": "o",
    "EconML_SLearner_Lasso": "^",
    "EconML_XLearner_Lasso": "d",
    "DiffPOLearner": "H",
    "Truth": "<",
}

datasets_names_map = {
    "tcga_100": "TCGA", 
    "twins": "Twins", 
    "news_100": "News", 
    "all_notupro_technologies": "AllNoTuproTechnologies",
    "all_notupro_technologies_small": "AllNoTuproTechnologiesSmall",
    "dummy_data": "DummyData",
    "selected_technologies_pategan_1000": "selected_technologies_pategan_1000",
    "selected_technologies_with_fastdrug": "selected_technologies_with_fastdrug",
    "cytof_normalized":"cytof_normalized",
    "cytof_normalized_with_fastdrug":"cytof_normalized_with_fastdrug",
    "cytof_pategan_1000_normalized": "cytof_pategan_1000_normalized",
    "all_notupro_technologies_with_fastdrug": "all_notupro_technologies_with_fastdrug",
    "acic": "ACIC2016", 
    "depmap_drug_screen_2_drugs": "depmap_drug_screen_2_drugs",
    "depmap_drug_screen_2_drugs_norm": "depmap_drug_screen_2_drugs_norm",
    "depmap_drug_screen_2_drugs_all_features": "depmap_drug_screen_2_drugs_all_features",
    "depmap_drug_screen_2_drugs_all_features_norm": "depmap_drug_screen_2_drugs_all_features_norm",
    "depmap_crispr_screen_2_kos": "depmap_crispr_screen_2_kos",
    "depmap_crispr_screen_2_kos_norm": "depmap_crispr_screen_2_kos_norm",
    "depmap_crispr_screen_2_kos_all_features": "depmap_crispr_screen_2_kos_all_features",
    "depmap_crispr_screen_2_kos_all_features_norm": "depmap_crispr_screen_2_kos_all_features_norm",
    "depmap_drug_screen_2_drugs_100pcs_norm":"depmap_drug_screen_2_drugs_100pcs_norm",
    "depmap_drug_screen_2_drugs_5000hv_norm":"depmap_drug_screen_2_drugs_5000hv_norm",
    "depmap_crispr_screen_2_kos_100pcs_norm":"depmap_crispr_screen_2_kos_100pcs_norm",
    "depmap_crispr_screen_2_kos_5000hv_norm":"depmap_crispr_screen_2_kos_5000hv_norm",
    "independent_normally_dist": "independent_normally_dist",
    "ovarian_semi_synthetic_l1":"ovarian_semi_synthetic_l1",
    "ovarian_semi_synthetic_rf":"ovarian_semi_synthetic_rf",
    "melanoma_semi_synthetic_l1": "melanoma_semi_synthetic_l1",
    "pred": "Predictive confounding", 
    "prog": "Prognostic confounding", 
    "irrelevant_var": "Non-confounded propensity",
    "selective": "General confounding"}

metric_names_map = {
    'Pred: Pred features ACC': r'Predictive $\mathrm{Attr}$', #^{\mathrm{pred}}_{\mathrm{pred}}$',
    'Pred: Prog features ACC': r'$\mathrm{Attr}^{\mathrm{pred}}_{\mathrm{prog}}$',
    'Pred: Select features ACC': r'$\mathrm{Attr}^{\mathrm{pred}}_{\mathrm{select}}$',
    'Prog: Pred features ACC': r'$\mathrm{Attr}^{\mathrm{prog}}_{\mathrm{pred}}$',
    'Prog: Prog features ACC': r'Prognostic $\mathrm{Attr}$', #^{\mathrm{prog}}$', #_{\mathrm{prog}}$',
    'Prog: Select features ACC': r'$\mathrm{Attr}^{\mathrm{prog}}_{\mathrm{select}}$',
    'Select: Pred features ACC': r'$\mathrm{Attr}^{\mathrm{select}}_{\mathrm{pred}}$',
    'Select: Prog features ACC': r'$\mathrm{Attr}^{\mathrm{select}}_{\mathrm{prog}}$',
    'Select: Select features ACC': r'$\mathrm{Attr}^{\mathrm{select}}_{\mathrm{select}}$',
    'CI Coverage': 'CI Coverage',
    'Normalized PEHE': 'N. PEHE',
    'PEHE': 'PEHE',
    'CF RMSE': 'CF-RMSE',
    'AUROC': 'AUROC',
    'Factual AUROC': 'Factual AUROC',
    'CF AUROC': "CF AUROC",
    'Factual RMSE': 'F-RMSE',
    "Factual RMSE Y0": "F-RMSE Y0", 
    "Factual RMSE Y1": "F-RMSE Y1",
    "CF RMSE Y0": "CF-RMSE Y0",
    "CF RMSE Y1": "CF-RMSE Y1",
    'Normalized F-RMSE': 'N. F-RMSE',
    'Normalized CF-RMSE': 'N. CF-RMSE',
    "F-Outcome true mean":"F-Outcome true mean",
    "CF-Outcome true mean":"CF-Outcome true mean",
    "F-Outcome true std":"F-Outcome true std",
    "CF-Outcome true std":"CF-Outcome true std",
    "F-CF Outcome Diff":"F-CF Outcome Diff",
    'Swap AUROC@1': 'AUROC@1',
    'Swap AUPRC@1': 'AUPRC@1',
    'Swap AUROC@5': 'AUROC@5',
    'Swap AUPRC@5': 'AUPRC@5',
    'Swap AUROC@tre': 'AUROC@tre',
    'Swap AUPRC@tre': 'AUPRC@tre',
    'Swap AUROC@all': 'AUROC',
    'Swap AUPRC@all': 'AUPRC',
    "GT Pred Expertise": r'$\mathrm{B}^{\pi}_{Y_1-Y_0}$',
    "GT Prog Expertise": r'$\mathrm{B}^{\pi}_{Y_0}$',
    "GT Tre Expertise": r'$\mathrm{B}^{\pi}_{Y_1}$',
    "Upd. GT Pred Expertise": r'$\mathrm{B}^{\hat{\pi}}_{Y_1-Y_0}$',
    "Upd. GT Prog Expertise": r'$\mathrm{B}^{\hat{\pi}}_{Y_0}$',
    "Upd. GT Tre Expertise": r'$\mathrm{B}^{\hat{\pi}}_{Y_1}$',
    "GT Expertise Ratio": r'$\mathrm{E}^{\pi}_{\mathrm{ratio}}$',
    "GT Total Expertise": r'$\mathrm{B}^{\pi}_{Y_0,Y_1}$',
    "ES Pred Expertise": "ES Pred Bias",
    "ES Prog Expertise": "ES Prog Bias",
    "ES Total Expertise": "ES Outcome Bias",
    "Pred Precision": r'$\mathrm{Prec}^{\hat{\pi}}_{\mathrm{Ass.}}$',
    "Policy Precision": r'$\mathrm{Prec}^{\pi}_{\mathrm{Ass.}}$',
    'T Distribution: Train': 'T Distribution: Train',
    'T Distribution: Test': 'T Distribution: Test',
    'True Swap Perc': 'True Swap Perc',
    "Normalized F-CF Diff": "Normalized F-CF Diff",
    'Training Duration': 'Training Duration',
    "FC PEHE":"PEHE(Model) - PEHE(TARNet)",
    "FC F-RMSE":"Rel. N. F-RMSE",
    "FC CF-RMSE":"Rel. N. CF-RMSE",
    "FC Swap AUROC":"Rel. AUROC",
    "FC Swap AUPRC":"Rel. AUPRC",
    "GT In-context Var":r'$\mathrm{B}^{\pi}_{X}$', 
    "ES In-context Var":"ES Total Bias",
    "GT-ES Pred Expertise Diff":r'$\mathrm{E}^{\pi}_{\mathrm{pred}}$ Error',
    "GT-ES Prog Expertise Diff":r'$\mathrm{E}^{\pi}_{\mathrm{prog}}$ Error',
    "GT-ES Total Expertise Diff":r'$\mathrm{E}^{\pi}$ Error',
    "RMSE Y0":"RMSE Y0",
    "RMSE Y1":"RMSE Y1",
}

learners_names_map = {
    "Torch_TLearner":"T-Learner-MLP", 
    "Torch_SLearner": "S-Learner-MLP", 
    "Torch_TARNet": "Baseline-TAR",  
    "Torch_DragonNet": "DragonNet-1 (Act. Pred.)",
    "Torch_DragonNet_2": "DragonNet-2 (Act. Pred.)",
    "Torch_DragonNet_4": "DragonNet-4 (Act. Pred.)",
    "Torch_DRLearner": "Direct-DR", 
    "Torch_XLearner": "XLearner-MLP (Direct)", 
    "Torch_CFRNet_0.001": 'CFRNet-0.001 (Balancing)', #-\gamma=0.001)$',  
    "Torch_CFRNet_0.01": 'CFRNet-0.01 (Balancing)', 
    "Torch_CFRNet_0.0001": 'CFRNet-0.0001 (Balancing)', 
    'Torch_ActionNet': "ActionNet (Act. Pred.)",
    "Torch_RALearner": "RA-Learner",
    "Torch_ULearner": "U-Learner",
    "Torch_PWLearner":"Torch_PWLearner",
    "Torch_RLearner":"Torch_RLearner",
    "Torch_FlexTENet":"Torch_FlexTENet",
    "EconML_CausalForestDML": "CausalForestDML",
    "EconML_DML": "APred-Prop-Lasso",
    "EconML_DMLOrthoForest": "DMLOrthoForest",
    "EconML_DRLearner": "DRLearner",
    "EconML_DROrthoForest": "DROrthoForest",
    "EconML_ForestDRLearner": "ForestDRLearner",
    "EconML_LinearDML": "LinearDML",
    "EconML_LinearDRLearner": "LinearDRLearner",
    "EconML_SparseLinearDML": "SparseLinearDML",
    "EconML_SparseLinearDRLearner": "SparseLinearDRLearner",
    "EconML_TLearner_Lasso": "T-Learner-Lasso",
    "EconML_SLearner_Lasso": "S-Learner-Lasso",
    "EconML_XLearner_Lasso": "XLearnerLasso",
    "DiffPOLearner": "DiffPOLearner",
    
    "Truth": "Truth"
}

compare_values_map = {
    # Propensity
    "none_prog": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{Y_0}}^\beta$',
    "none_tre": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{Y_1}}^\beta$',
    "none_pred": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{Y_1-Y_0}}^\beta$',
    "rct_none": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{X_{irr}}}^\beta$',
    "none_pred_overlap": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{X_{pred}}}^\beta$',

    # Toy
    "toy7": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{T_7}}^\beta$',
    "toy8_nonlinear": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{T_8}}^\beta$',
    "toy1_linear": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{T_1^{lin}}}^\beta$',
    "toy1_nonlinear": r'$\mathrm{Toy 1:} \pi_{\mathrm{T_1}}^\beta$', #r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{T_1}}^\beta$',
    "toy2_linear": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{T_3^{lin}}}^\beta$',
    "toy2_nonlinear": r'$\mathrm{Toy 3:} \pi_{\mathrm{T_3}}^\beta$',
    "toy3_nonlinear": r'$\mathrm{Toy 2:} \pi_{\mathrm{T_2}}^\beta$',
    "toy4_nonlinear": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{T_4}}^\beta$',
    "toy5": r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{T_5}}^\beta$',
    "toy6_nonlinear": r'$\mathrm{Toy 4:} \pi_{\mathrm{T_4}}^\beta$', #r'$\pi_{\mathrm{RCT}} \rightarrow \pi_{\mathrm{T_6}}^\beta$',

    # Expertise
    # "prog_tre": r'$\pi_{\mathrm{Y_0}}^{\beta=4} \rightarrow \pi_{\mathrm{Y_1-Y_0}}^{\beta=4} \rightarrow \pi_{\mathrm{Y_1}}^{\beta=4}$',
    # "none_prog": r'$\pi_{\mathrm{X_{rand}}}^{\beta=4} \rightarrow \pi_{\mathrm{Y_0}}^{\beta=4}$',
    # "none_tre": r'$\pi_{\mathrm{X_{rand}}}^{\beta=4} \rightarrow \pi_{\mathrm{Y_1}}^{\beta=4}$',
    # "none_pred": r'$\pi_{\mathrm{X_{rand}}}^{\beta=4} \rightarrow \pi_{\mathrm{Y_1-Y_0}}^{\beta=4}$',
    #["prog_tre", "none_prog", "none_tre", "none_pred"]

    0: r'$\pi_{\mathrm{RCT}}$',
    2: r'$\pi_{\mathrm{Y_1-Y_0}}^{\beta=2}$',
    100: r'$\pi_{\mathrm{Y_1-Y_0}}^{\beta=100}$',

    "0": r'$\pi_{\mathrm{RCT}}$',
    "2": r'$\pi_{\mathrm{Y_1-Y_0}}^{\beta=2}$',
    "100": r'$\pi_{\mathrm{Y_1-Y_0}}^{\beta=100}$',
}


def plot_results_datasets_compare(results_df: pd.DataFrame, 
                          model_names: list,
                          dataset: str,
                          compare_axis: str,
                          compare_axis_values,
                          x_axis, 
                          x_label_name, 
                          x_values_to_plot, 
                          metrics_list, 
                          learners_list, 
                          figsize, 
                          legend_position, 
                          seeds_list, 
                          n_splits,
                          sharey=False, 
                          legend_rows=1,
                          dim_X=1,
                          log_x_axis = False): 
    """
    Plot the results for a given dataset.
    """
    # Get the unique values of the compare axis
    if compare_axis_values is None:
        compare_axis_values = results_df[compare_axis].unique()

    metrics_list = ["Pred Precision"]
    # Initialize the plot
    nrows = len(metrics_list)
    columns = len(compare_axis_values)
    figsize = (3*columns+2, 3*nrows)
    #figsize = (3*columns, 3)

    font_size=10
    fig, axs = plt.subplots(len(metrics_list), len(compare_axis_values), figsize=figsize, squeeze=False, sharey=sharey, dpi=500)
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # Aggregate results across seeds for each metric
    for i in range(len(compare_axis_values)):
        cmp_value = compare_axis_values[i]
        for metric_id, metric in enumerate(metrics_list):
            for model_name in model_names:
                # Extract results for individual cate models
                sub_df = results_df.loc[(results_df["Learner"] == model_name)]
                sub_df = sub_df.loc[(sub_df[compare_axis] == cmp_value)][[x_axis, metric]]
                sub_df = sub_df[sub_df[x_axis].isin(x_values_to_plot)]
                sub_df_mean = sub_df.groupby(x_axis).agg('median').reset_index()
                sub_df_std = sub_df.groupby(x_axis).agg('std').reset_index()
                sub_df_min = sub_df.groupby(x_axis).agg('min').reset_index()
                sub_df_max = sub_df.groupby(x_axis).agg('max').reset_index()

                # Plot the results
                x_values = sub_df_mean.loc[:, x_axis].values

                try:
                    y_values = sub_df_mean.loc[:, metric].values
                except:
                    continue

                y_err = sub_df_std.loc[:, metric].values / (np.sqrt(n_splits*len(seeds_list)))
                y_min = sub_df_min.loc[:, metric].values
                y_max = sub_df_max.loc[:, metric].values
                
                # axs[metric_id][i].plot(x_values, y_values, label=learners_names_map[model_name], 
                #                                             color=learner_colors[model_name], linestyle=learner_linestyles[model_name], marker=learner_markers[model_name], markersize=5)
                axs[metric_id][i].plot(x_values, y_values, label=learners_names_map[model_name], 
                                                            color=learner_colors[model_name], linestyle=learner_linestyles[model_name], marker=learner_markers[model_name], markersize=3, alpha=0.5)
                axs[metric_id][i].fill_between(x_values, y_values-y_err, y_values+y_err, alpha=0.1, color=learner_colors[model_name])

            
            
            # if log_x_axis:
            #     axs[metric_id][i].set_xscale('symlog', linthresh=0.5, base=2)
            #     #axs[metric_id][i].fill_between(x_values, y_min, y_max, alpha=0.1, color=learner_colors[model_name])
            
            axs[metric_id][i].tick_params(axis='x', labelsize=font_size-2)
            axs[metric_id][i].tick_params(axis='y', labelsize=font_size-1)

            
            axs[metric_id][i].set_title(compare_values_map[cmp_value], fontsize=font_size+11, y=1.04)

            axs[metric_id][i].set_xlabel(x_label_name, fontsize=font_size-1)
            if i == 0:
                axs[metric_id][i].set_ylabel(metric_names_map[metric], fontsize=font_size-1)

            if log_x_axis:
                axs[metric_id][i].set_xscale('symlog', linthresh=0.5, base=2)
                # Display as fractions if not integers and as integers if integers
                # axs[0][i].xaxis.set_major_formatter(ScalarFormatter())
                # Get the current ticks
                current_ticks = axs[metric_id][i].get_xticks()
                
                # Calculate the midpoint between the first and second tick
                if len(current_ticks) > 1:
                    midpoint = (current_ticks[0] + current_ticks[1]) / 2
                    # Add the midpoint to the list of ticks
                    new_ticks = [current_ticks[0], midpoint] + list(current_ticks[1:])
                    axs[metric_id][i].set_xticks(new_ticks)

                # Add a tick at 0.25
                axs[metric_id][i].set_xticks(sorted(set(axs[metric_id][i].get_xticks()).union({0.25})))
                axs[metric_id][i].xaxis.set_major_formatter(FuncFormatter(custom_formatter))

            if metric in ["True Swap Perc", "T Distribution: Train", "T Distribution: Test", "GT Total Expertise", "ES Total Expertise", "GT Expertise Ratio", "GT Pred Expertise", "GT Prog Expertise", "ES Pred Expertise", "ES Prog Expertise","GT In-context Var","ES In-context Var","GT-ES Pred Expertise Diff","GT-ES Prog Expertise Diff","GT-ES Total Expertise Diff", "Policy Precision", "GT In-context Var", "GT Total Expertise", "GT Prog Expertise", "GT Tre Expertise", "GT Pred Expertise", "Upd. GT Prog Expertise", "Upd. GT Tre Expertise", "Upd. GT Pred Expertise"]:
                axs[metric_id][i].set_ylim(0, 1)

            if metric == "PEHE":
                axs[metric_id][i].set_ylim(top = 1.75)
            #axs[metric_id][i].set_ylim(bottom=0.475)
            #axs[metric_id][i].set_aspect(0.7/axs[metric_id][i].get_data_ratio(), adjustable='box')
            #axs[metric_id][i].tick_params(axis='y', labelsize=font_size-1)

            # axs[metric_id][i].tick_params(
            #     axis='x',          # changes apply to the x-axis
            #     which='both',      # both major and minor ticks are affected
            #     bottom=False,      # ticks along the bottom edge are off
            #     top=False,         # ticks along the top edge are off
            #     labelbottom=False) # labels along the bottom edge are off

    # Add the legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend_rows = 6

    # Iterate over each row of subplots
    for row in range(len(axs)):
        # Create a legend for each row
        handles, labels = axs[row, -1].get_legend_handles_labels()
        axs[row, -1].legend(
            lines[:len(learners_list)],
            labels[:len(learners_list)],
            ncol=1, #len(learners_list) if legend_rows == 1 else int((len(learners_list) + 1) / legend_rows),
            loc='center right',
            bbox_to_anchor=(1.8, 0.5),
            prop={'size': font_size+2}
        )


    #fig.tight_layout()
    plt.subplots_adjust( wspace=0.07)
    return fig   

def plot_performance_metrics(results_df: pd.DataFrame, 
                          model_names: list,
                          dataset: str,
                          compare_axis: str,
                          compare_axis_values,
                          x_axis, 
                          x_label_name, 
                          x_values_to_plot, 
                          metrics_list, 
                          learners_list, 
                          figsize, 
                          legend_position, 
                          seeds_list, 
                          n_splits,
                          sharey=False, 
                          legend_rows=1,
                          dim_X=1,
                          log_x_axis = False): 
    
    # Get the unique values of the compare axis
    if compare_axis_values is None:
        compare_axis_values = results_df[compare_axis].unique()

    metrics_list = ['PEHE', 'FC PEHE', "Pred Precision", 'Pred: Pred features ACC', 'Prog: Prog features ACC'] #]
    #log_x_axis=False
    # Initialize the plot
    nrows = len(metrics_list)
    columns = len(compare_axis_values)

    #figsize = (3*columns+2, 3*nrows) #PREV
    figsize = (3*columns+2, 3.4*nrows)
    #figsize = (3*columns, 3)

    font_size=10
    fig, axs = plt.subplots(len(metrics_list), len(compare_axis_values), figsize=figsize, squeeze=False, sharey=sharey, dpi=500)
    plt.gcf().subplots_adjust(bottom=0.15)
    
    model_names_cpy = model_names.copy()
    # Aggregate results across seeds for each metric
    for i in range(len(compare_axis_values)):
        cmp_value = compare_axis_values[i]
        for metric_id, metric in enumerate(metrics_list):
            # if metric in ["FC PEHE", 'Prog: Prog features ACC', 'Prog: Pred features ACC']:
            #     model_names = ["Torch_TARNet","Torch_DragonNet","Torch_CFRNet_0.001","EconML_TLearner_Lasso"]
            # else:
            model_names = model_names_cpy #["Torch_TARNet","Torch_DragonNet","Torch_ActionNet", "Torch_CFRNet_0.001","EconML_TLearner_Lasso"]

            for model_name in model_names:
                # Extract results for individual cate models
                sub_df = results_df.loc[(results_df["Learner"] == model_name)]
                sub_df = sub_df.loc[(sub_df[compare_axis] == cmp_value)][[x_axis, metric]]
                sub_df = sub_df[sub_df[x_axis].isin(x_values_to_plot)]
                sub_df_mean = sub_df.groupby(x_axis).agg('median').reset_index()
                sub_df_std = sub_df.groupby(x_axis).agg('std').reset_index()
                sub_df_min = sub_df.groupby(x_axis).agg('min').reset_index()
                sub_df_max = sub_df.groupby(x_axis).agg('max').reset_index()

                # Plot the results
                x_values = sub_df_mean.loc[:, x_axis].values

                try:
                    y_values = sub_df_mean.loc[:, metric].values
                except:
                    continue

                y_err = sub_df_std.loc[:, metric].values / (np.sqrt(n_splits*len(seeds_list)))
                y_min = sub_df_min.loc[:, metric].values
                y_max = sub_df_max.loc[:, metric].values
                
                # axs[metric_id][i].plot(x_values, y_values, label=learners_names_map[model_name], 
                #                                             color=learner_colors[model_name], linestyle=learner_linestyles[model_name], marker=learner_markers[model_name], markersize=5)
                axs[metric_id][i].plot(x_values, y_values, label=learners_names_map[model_name], 
                                                            color=learner_colors[model_name], linestyle=learner_linestyles[model_name], marker=learner_markers[model_name], markersize=3, alpha=0.5)
                axs[metric_id][i].fill_between(x_values, y_values-y_err, y_values+y_err, alpha=0.1, color=learner_colors[model_name])

            
            
            # if log_x_axis:
            #     axs[metric_id][i].set_xscale('symlog', linthresh=0.5, base=2)
            #     #axs[metric_id][i].fill_between(x_values, y_min, y_max, alpha=0.1, color=learner_colors[model_name])
            
            axs[metric_id][i].tick_params(axis='x', labelsize=font_size-2)
            axs[metric_id][i].tick_params(axis='y', labelsize=font_size-1)

            
            if metric_id == 0:
                axs[metric_id][i].set_title(compare_values_map[cmp_value], fontsize=font_size+1, y=1.0)


            axs[metric_id][i].set_xlabel(x_label_name, fontsize=font_size-1)
            if i == 0:
                axs[metric_id][i].set_ylabel(metric_names_map[metric], fontsize=font_size-1)

            if log_x_axis:
                axs[metric_id][i].set_xscale('symlog', linthresh=0.5, base=2)
                # Display as fractions if not integers and as integers if integers
                # axs[0][i].xaxis.set_major_formatter(ScalarFormatter())
                # Get the current ticks
                current_ticks = axs[metric_id][i].get_xticks()
                
                # Calculate the midpoint between the first and second tick
                if len(current_ticks) > 1:
                    midpoint = (current_ticks[0] + current_ticks[1]) / 2
                    # Add the midpoint to the list of ticks
                    new_ticks = [current_ticks[0], midpoint] + list(current_ticks[1:])
                    axs[metric_id][i].set_xticks(new_ticks)

                # Add a tick at 0.25
                axs[metric_id][i].set_xticks(sorted(set(axs[metric_id][i].get_xticks()).union({0.25})))
                axs[metric_id][i].xaxis.set_major_formatter(FuncFormatter(custom_formatter))

            if metric in ["True Swap Perc", "T Distribution: Train", "T Distribution: Test", "GT Total Expertise", "ES Total Expertise", "GT Expertise Ratio", "GT Pred Expertise", "GT Prog Expertise", "ES Pred Expertise", "ES Prog Expertise","GT In-context Var","ES In-context Var","GT-ES Pred Expertise Diff","GT-ES Prog Expertise Diff","GT-ES Total Expertise Diff", "Policy Precision", "GT In-context Var", "GT Total Expertise", "GT Prog Expertise", "GT Tre Expertise", "GT Pred Expertise", "Upd. GT Prog Expertise", "Upd. GT Tre Expertise", "Upd. GT Pred Expertise"]:
                axs[metric_id][i].set_ylim(0, 1)

            # if metric == "PEHE":
            #     axs[metric_id][i].set_ylim(top = 1.75)
            #axs[metric_id][i].set_ylim(bottom=0.475)
            #axs[metric_id][i].set_aspect(0.7/axs[metric_id][i].get_data_ratio(), adjustable='box')
            #axs[metric_id][i].tick_params(axis='y', labelsize=font_size-1)

            axs[metric_id][i].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

    # Add the legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend_rows = 6

    # Iterate over each row of subplots
    for row in range(len(axs)):
        # Create a legend for each row
        handles, labels = axs[row, -1].get_legend_handles_labels()
        axs[row, -1].legend(
            lines[:len(learners_list)],
            labels[:len(learners_list)],
            ncol=1, #len(learners_list) if legend_rows == 1 else int((len(learners_list) + 1) / legend_rows),
            loc='center right',
            bbox_to_anchor=(1.9, 0.5),
            prop={'size': font_size+2}
        )


    plt.subplots_adjust( wspace=0.07)
    #fig.tight_layout()
    return fig  



def plot_performance_metrics_f_cf(results_df: pd.DataFrame, 
                          model_names: list,
                          dataset: str,
                          compare_axis: str,
                          compare_axis_values,
                          x_axis, 
                          x_label_name, 
                          x_values_to_plot, 
                          metrics_list, 
                          learners_list, 
                          figsize, 
                          legend_position, 
                          seeds_list, 
                          n_splits,
                          sharey=False, 
                          legend_rows=1,
                          dim_X=1,
                          log_x_axis = False): 
    # Get the unique values of the compare axis
    if compare_axis_values is None:
        compare_axis_values = results_df[compare_axis].unique()

    # Initialize the plot
    #model_names = model_names[0] #["Torch_TARNet"] #[EconML_TLearner_Lasso"]
    columns = len(compare_axis_values)
    rows = len(model_names)
    figsize = (3*columns+2, 3.3*rows)
    #figsize = (3*columns, 3)
    font_size=10
    fig, axs = plt.subplots(len(model_names), len(compare_axis_values), figsize=figsize, squeeze=False, sharey=sharey, dpi=500)
    #plt.gcf().subplots_adjust(bottom=0.15)
    
    # Filter results_df for first model and first seed and first split
    
    # results_df = results_df.loc[(results_df["Seed"] == seeds_list[0])]
    # results_df = results_df.loc[(results_df["Split ID"] == 0)]

    # Only consider expertise metrics
    #colors = ['black', 'orange', 'darkorange', 'orchid', 'darkorchid']
    colors = ['blue', 'lightcoral', 'lightgreen', 'red', 'green']

    markers = ['o', 'D', 'D', 'x',  'x']
    metrics_list = ["PEHE", "Factual RMSE Y0", "Factual RMSE Y1", "CF RMSE Y0", "CF RMSE Y1"]
    filtered_df = results_df[[x_axis, compare_axis] + metrics_list]

    # Aggregate results across seeds for each metric
    for model_id, model_name in enumerate(model_names):
        filtered_df_model = filtered_df.loc[(results_df["Learner"] == model_name)]
        for i in range(len(compare_axis_values)):
            cmp_value = compare_axis_values[i]

            # Plot all metric outcomes as lines in a single plot for the given cmp_value and use x_axis as x-axis
            x_values = filtered_df_model[x_axis].values

            for metric_id, metric in enumerate(metrics_list):
                # Extract results for individual cate models

                sub_df = filtered_df_model.loc[(filtered_df_model[compare_axis] == cmp_value)][[x_axis, metric]]
                sub_df = sub_df[sub_df[x_axis].isin(x_values_to_plot)]
                sub_df_mean = sub_df.groupby(x_axis).agg('median').reset_index()
                sub_df_std = sub_df.groupby(x_axis).agg('std').reset_index()
                sub_df_min = sub_df.groupby(x_axis).agg('min').reset_index()
                sub_df_max = sub_df.groupby(x_axis).agg('max').reset_index()

                # Plot the results
                x_values = sub_df_mean.loc[:, x_axis].values

                try:
                    y_values = sub_df_mean.loc[:, metric].values
                except:
                    continue

                y_err = sub_df_std.loc[:, metric].values / (np.sqrt(n_splits*len(seeds_list)))
                y_min = sub_df_min.loc[:, metric].values
                y_max = sub_df_max.loc[:, metric].values

                # use a different linestyle for each metric

                axs[model_id][i].plot(x_values, y_values, label=metric_names_map[metric], 
                                                                color=colors[metric_id], linestyle='-', marker=markers[metric_id], alpha=0.5, markersize=3)
                axs[model_id][i].fill_between(x_values, y_values-y_err, y_values+y_err, alpha=0.1, color=colors[metric_id])


                axs[model_id][i].tick_params(axis='x', labelsize=font_size-2)
                axs[model_id][i].tick_params(axis='y', labelsize=font_size-1)
                
                axs[model_id][i].set_title(compare_values_map[cmp_value], fontsize=font_size+2, y=1.01)

                axs[model_id][i].set_xlabel(x_label_name, fontsize=font_size-1)

            # if i == 0:
            #     axs[model_id][i].set_ylabel(metric_names_map[metric], fontsize=font_size-1)

            if log_x_axis:
                axs[model_id][i].set_xscale('symlog', linthresh=0.5, base=2)
                # Display as fractions if not integers and as integers if integers
                # axs[0][i].xaxis.set_major_formatter(ScalarFormatter())
                # Get the current ticks
                current_ticks = axs[model_id][i].get_xticks()
                
                # Calculate the midpoint between the first and second tick
                if len(current_ticks) > 1:
                    midpoint = (current_ticks[0] + current_ticks[1]) / 2
                    # Add the midpoint to the list of ticks
                    new_ticks = [current_ticks[0], midpoint] + list(current_ticks[1:])
                    axs[model_id][i].set_xticks(new_ticks)

                # Add a tick at 0.25
                axs[model_id][i].set_xticks(sorted(set(axs[model_id][i].get_xticks()).union({0.25})))
                axs[model_id][i].xaxis.set_major_formatter(FuncFormatter(custom_formatter))
            axs[model_id][i].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            
            axs[model_id][i].tick_params(axis='y', labelsize=font_size-1)
            #axs[model_id][i].set_aspect(0.7/axs[model_id][i].get_data_ratio(), adjustable='box')


    # Add the legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # Add legends to the right of each row
    for i, row in enumerate(axs):
        lines_labels = [ax.get_legend_handles_labels() for ax in row]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        row[-1].legend(
            lines[:len(metrics_list)],
            labels[:len(metrics_list)],
            loc='center right',
            bbox_to_anchor=(1.8, 0.5),
            ncol=1,
            prop={'size': font_size+2},
            title_fontsize=font_size+4,
            title=learners_names_map[model_names[i]]
        )

    #fig.tight_layout()
    plt.subplots_adjust( wspace=0.07)

    return fig


def plot_expertise_metrics(results_df: pd.DataFrame, 
                          model_names: list,
                          dataset: str,
                          compare_axis: str,
                          compare_axis_values,
                          x_axis, 
                          x_label_name, 
                          x_values_to_plot, 
                          metrics_list, 
                          learners_list, 
                          figsize, 
                          legend_position, 
                          seeds_list, 
                          n_splits,
                          sharey=False, 
                          legend_rows=1,
                          dim_X=1,
                          log_x_axis = False): 
    
    if compare_axis_values is None:
        compare_axis_values = results_df[compare_axis].unique()

    # Initialize the plot
    columns = len(compare_axis_values)
    figsize = (3*columns+2, 3)
    font_size=10
    fig, axs = plt.subplots(1, len(compare_axis_values), figsize=figsize, squeeze=False, sharey=sharey, dpi=500)
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # Filter results_df for first model and first seed and first split
    results_df = results_df.loc[(results_df["Learner"] == model_names[0])]
    results_df = results_df.loc[(results_df["Seed"] == seeds_list[0])]
    results_df = results_df.loc[(results_df["Split ID"] == 0)]


    # Only consider expertise metrics
    colors = ['black', 'grey', 'red', 'green', 'blue']
    markers = ['o', 'x', 'x',  'x', 'x']
    metrics_list = ["Policy Precision", "GT In-context Var", "GT Prog Expertise", "GT Tre Expertise", "GT Pred Expertise"]
    sub_df = results_df[[x_axis, compare_axis, "Seed", "Split ID"] + metrics_list]

    # Aggregate results across seeds for each metric
    for i in range(len(compare_axis_values)):
        cmp_value = compare_axis_values[i]

        # Plot all metric outcomes as lines in a single plot for the given cmp_value and use x_axis as x-axis
        filtered_df = sub_df[(sub_df[compare_axis] == cmp_value)]
        x_values = filtered_df[x_axis].values

        for metric_id, metric in enumerate(metrics_list):
            y_values = filtered_df[metric].values
            # use a different linestyle for each metric

            axs[0][i].plot(x_values, y_values, label=metric_names_map[metric], color=colors[metric_id], linestyle='-', marker=markers[metric_id], alpha=0.5,  markersize=5)
            

            # if i == 0:
            #     axs[0][i].set_ylabel("Selection Bias", fontsize=font_size)

            axs[0][i].tick_params(axis='x', labelsize=font_size-2)
            axs[0][i].tick_params(axis='y', labelsize=font_size-1)
            axs[0][i].set_title(compare_values_map[cmp_value], fontsize=font_size+11, y=1.04)
            axs[0][i].set_xlabel(x_label_name, fontsize=font_size-1)

        if log_x_axis:
            axs[0][i].set_xscale('symlog', linthresh=0.5, base=2)
            # Display as fractions if not integers and as integers if integers
            # axs[0][i].xaxis.set_major_formatter(ScalarFormatter())
            # Get the current ticks
            current_ticks = axs[0][i].get_xticks()
            
            # Calculate the midpoint between the first and second tick
            if len(current_ticks) > 1:
                midpoint = (current_ticks[0] + current_ticks[1]) / 2
                # Add the midpoint to the list of ticks
                new_ticks = [current_ticks[0], midpoint] + list(current_ticks[1:])
                axs[0][i].set_xticks(new_ticks)

            # Add a tick at 0.25
            axs[0][i].set_xticks(sorted(set(axs[0][i].get_xticks()).union({0.25})))
            axs[0][i].xaxis.set_major_formatter(FuncFormatter(custom_formatter))

        # if i == 0:
        #         axs[0][i].set_ylabel(metric_names_map[metric], fontsize=font_size-1)

        axs[0][i].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        #axs[0][i].set_aspect(0.7/axs[0][i].get_data_ratio(), adjustable='box')
        axs[0][i].tick_params(axis='y', labelsize=font_size-1)

    # Add the legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    fig.legend(
        lines[:len(metrics_list)],
        labels[:len(metrics_list)],
        loc='center right',  # Position the legend to the right
        #bbox_to_anchor=(1, 0.5),  # Adjust the anchor point to the right center
        ncol=1,  # Set the number of columns to 1 for a vertical legend
        prop={'size': font_size+2}
    )

    #fig.tight_layout()
    plt.subplots_adjust( wspace=0.07)

    return fig
    
                           

def merge_pngs(images, axis="horizontal"):
    """
    Merge a list of png images into a single image.
    """
    widths, heights = zip(*(i.size for i in images))

    if axis == "vertical":
        total_height = sum(heights)
        max_width = max(widths)

        new_im = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0,y_offset))
            y_offset += im.size[1]
        
        return new_im
    
    elif axis == "horizontal":
        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        
        return new_im
    
