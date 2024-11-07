import argparse
import sys
import os
from typing import Any
import pandas as pd 
import src.iterpretability.logger as log
from src.iterpretability.experiments.sample_size_sensitivity import SampleSizeSensitivity
from src.iterpretability.experiments.important_feature_num_sensitivity import ImportantFeatureNumSensitivity
from src.iterpretability.experiments.propensity_scale_sensitivity import PropensityScaleSensitivity
from src.iterpretability.experiments.predictive_scale_sensitivity import PredictiveScaleSensitivity
from src.iterpretability.experiments.treatment_space_sensitivity import TreatmentSpaceSensitivity
from src.iterpretability.experiments.sample_size_sensitivity_visualization import SampleSizeSensitivityVisualization
from src.iterpretability.experiments.data_dimensionality_sensitivity import DataDimensionalitySensitivity
from src.iterpretability.experiments.feature_overlap_sensitivity import FeatureOverlapSensitivity
from src.iterpretability.experiments.expertise_sensitivity import ExpertiseSensitivity
from src.iterpretability.simulators import TYSimulator, TSimulator
from src.iterpretability.datasets.data_loader import load

# Hydra for configuration
import hydra
from omegaconf import DictConfig, OmegaConf

def plot_result(cfg, fig_type="all", compare_axis_values=None):
    if cfg.experiment_name == "sample_size_sensitivity":
        exp = SampleSizeSensitivity(cfg)

    elif cfg.experiment_name == "important_feature_num_sensitivity":
        exp = ImportantFeatureNumSensitivity(cfg)

    elif cfg.experiment_name == "propensity_scale_sensitivity":
        exp = PropensityScaleSensitivity(cfg)

    elif cfg.experiment_name == "predictive_scale_sensitivity":
        if cfg.simulator.simulation_type == "T":
            print("PredictiveScaleSensitivity is not supported for TSimulator")
            return None
        
        exp = PredictiveScaleSensitivity(cfg)

    elif cfg.experiment_name == "treatment_space_sensitivity":
        if cfg.simulator.simulation_type == "T":
            print("PredictiveScaleSensitivity is not supported for TSimulator")
            return None
        
        exp = TreatmentSpaceSensitivity(cfg)

    elif cfg.experiment_name == "sample_size_sensitivity_visualization":
        if cfg.simulator.dim_Y != 1 or cfg.simulator.num_T != 2:
            print("SampleSizeSensitivityVisualization is only supported for dim_Y=1 and dim_T=2")
            return None
        
        if cfg.simulator.simulation_type == "T":
            print("Make sure to only pick two treatments for this to work. And remove this if statement.")
            return None
        
        exp = SampleSizeSensitivityVisualization(cfg)

    elif cfg.experiment_name == "data_dimensionality_sensitivity":
        exp = DataDimensionalitySensitivity(cfg)

    elif cfg.experiment_name == "feature_overlap_sensitivity":
        if cfg.simulator.simulation_type == "T":
            print("FeatureOverlapSensitivity is not supported for TSimulator")
            return None
        
        exp = FeatureOverlapSensitivity(cfg)
        
    elif cfg.experiment_name == "expertise_sensitivity":
        exp = ExpertiseSensitivity(cfg)

    else:
        raise ValueError(f"Invalid experiment name: {cfg.experiment_name}")

    exp.load_and_plot_results(fig_type=fig_type, compare_axis_values=compare_axis_values)


@hydra.main(config_path="conf", config_name="config_TY_tcga", version_base=None)
def main(cfg: DictConfig):
    ############################
    # 1. Plotting
    ############################
    # Set logging level
    log.add(sink=sys.stderr, level=cfg.log_level)

    for root_out, dirs_out, _ in os.walk(cfg.results_path):
        if os.path.exists(root_out) and not os.listdir(root_out):
            os.rmdir(root_out)
            continue
        
        for dir_out in dirs_out:
            if dir_out.startswith("archive"): # or not dir == "data_dimensionality_sensitivity":
                continue
            for root,dirs,_ in os.walk(os.path.join(root_out, dir_out)):
                for dir in dirs:
                    if os.path.exists(os.path.join(root, dir)) and not os.listdir(os.path.join(root, dir)):
                        os.rmdir(os.path.join(root, dir))
                        continue
                    
                    # create plots dir if it does not exist
                    if not os.path.exists(os.path.join(root, dir, "plots")):
                        os.makedirs(os.path.join(root, dir, "plots"))
                        
                    for _,_,files in os.walk(os.path.join(root, dir)):
                        for file in files:
                            if file.endswith(".yaml"): # and file.startswith("09_08_control_full_tcga_T2_Y1_TYsim_numbin0"): #"31_07_spo_depmap_crispr_screen_2_kos_T2_Y1_Tsim_numbin0"):
                                cfg = OmegaConf.load(os.path.join(root, dir, file))
                                cfg.metrics_to_plot = ["Policy Precision", "Pred Precision", "GT In-context Var", "GT Total Expertise", "GT Prog Expertise", "GT Tre Expertise", "GT Pred Expertise", "RMSE Y0",  "RMSE Y1", "PEHE", "Upd. GT Prog Expertise", "Upd. GT Tre Expertise", "Upd. GT Pred Expertise", "Factual RMSE Y0", "CF RMSE Y0", "Factual RMSE Y1", "CF RMSE Y1", "Factual RMSE", "CF RMSE", 'Normalized F-RMSE', 'Normalized CF-RMSE', 'Normalized PEHE', 'Swap AUROC@all', 'Swap AUPRC@all', "FC PEHE", "FC CF-RMSE", "FC Swap AUROC", "FC Swap AUPRC", 'Pred: Pred features ACC', 'Pred: Prog features ACC', 'Prog: Prog features ACC', 'Prog: Pred features ACC', "GT Expertise Ratio", "GT-ES Pred Expertise Diff", "GT-ES Prog Expertise Diff", "GT-ES Total Expertise Diff", 'T Distribution: Train', 'T Distribution: Test', 'Training Duration']
                                plots_folder = "plots/"

                                compare_axis_values = None
                                # if cfg.experiment_name == "propensity_scale_sensitivity" and "toy1_nonlinear" in cfg.propensity_types:
                                #     #compare_axis_values = [i for i in cfg.propensity_types if i != "toy5"]
                                #     compare_axis_values = ["toy1_nonlinear", "toy3_nonlinear", "toy2_nonlinear", "toy6_nonlinear"]

                                # cfg.repo_path = "/home/mike/UZH_USZ"
                                # cfg.results_path = "${repo_path}/code_mv/data_simulation/results"

                                try:
                                    cfg.plot_name_prefix = plots_folder+"bias"
                                    plot_result(cfg, fig_type="expertise", compare_axis_values=compare_axis_values)
                                except Exception as e:
                                    print("Error:", e)


                                try:
                                    cfg.plot_name_prefix = plots_folder+"pehe_and_rmse_per_model"
                                    plot_result(cfg, fig_type="performance_f_cf", compare_axis_values=compare_axis_values)
                                except Exception as e:
                                    print("Error:", e)

                                    
                                try:
                                    #cfg.model_names = ["EconML_TLearner_Lasso", "Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_DragonNet", "Torch_DragonNet_4"] #"Torch_ActionNet", "Torch_SLearner"]
                                    cfg.plot_name_prefix = plots_folder+"model_performance_comparison"
                                    plot_result(cfg, fig_type="performance", compare_axis_values=compare_axis_values)
                                except Exception as e:
                                    print("Error:", e)

                                # try:
                                #     #cfg.model_names = ["EconML_TLearner_Lasso", "Torch_TLearner", "Torch_SLearner", "Torch_TARNet", "Torch_XLearner", "Torch_CFRNet_0.001", "Torch_DragonNet", "Torch_ActionNet"] #"Torch_ActionNet", "Torch_SLearner"]
                                #     #cfg.model_names = ["EconML_TLearner_Lasso", "Torch_TLearner", "Torch_XLearner", "Torch_CFRNet_0.001", "Torch_DragonNet"] #"Torch_ActionNet", "Torch_SLearner"]
                                #     cfg.plot_name_prefix = plots_folder+"prec_all_models"
                                #     plot_result(cfg, compare_axis_values=compare_axis_values)
                                # except Exception as e:
                                #     print("Error:", e)

                                # try:
                                #     cfg.plot_name_prefix = plots_folder+"v1_no_S_Action"
                                #     plot_result(cfg, compare_axis_values=compare_axis_values)
                                # except Exception as e:
                                #     print("Error:", e)

                                # try:
                                #     cfg.model_names = ["Torch_TARNet", "Torch_DragonNet","Torch_DragonNet_2", "Torch_DragonNet_4", "Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #, "Torch_TLearner", "Torch_XLearner", "EconML_TLearner_Lasso", "Torch_ActionNet", "Torch_SLearner"]
                                #     cfg.plot_name_prefix = plots_folder+"v2_only_expertise_paper"
                                #     plot_result(cfg, compare_axis_values=compare_axis_values)
                                # except Exception as e:
                                #     print("Error:", e)

                                # try:
                                #     cfg.model_names = ["Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #["Torch_TARNet", "Torch_DragonNet", "Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #, "Torch_TLearner", "Torch_XLearner", "EconML_TLearner_Lasso", "Torch_ActionNet", "Torch_SLearner"]
                                #     cfg.plot_name_prefix = plots_folder+"v3_only_balancing"
                                #     plot_result(cfg, compare_axis_values=compare_axis_values)
                                # except Exception as e:
                                #     print("Error:", e)

                                # try:
                                #     cfg.model_names = ["Torch_TLearner", "Torch_SLearner", "EconML_TLearner_Lasso", "EconML_SLearner_Lasso"] #["Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #["Torch_TARNet", "Torch_DragonNet", "Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #, "Torch_TLearner", "Torch_XLearner", "EconML_TLearner_Lasso", "Torch_ActionNet", "Torch_SLearner"]
                                #     cfg.plot_name_prefix = plots_folder+"v4_torch_vs_econml"
                                #     plot_result(cfg, compare_axis_values=compare_axis_values)
                                # except Exception as e:
                                #     print("Error:", e)

                                # try:
                                #     cfg.model_names = ["EconML_TLearner_Lasso"] #["Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #["Torch_TARNet", "Torch_DragonNet", "Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #, "Torch_TLearner", "Torch_XLearner", "EconML_TLearner_Lasso", "Torch_ActionNet", "Torch_SLearner"]
                                #     cfg.plot_name_prefix = plots_folder+"v5_only_TLearner_Lasso"
                                #     plot_result(cfg, compare_axis_values=compare_axis_values)
                                # except Exception as e:
                                #     print("Error:", e)

                                # try:
                                #     cfg.model_names = ["EconML_TLearner_Lasso", "Torch_TLearner", "Torch_XLearner", "Torch_CFRNet_0.001", "Torch_DragonNet"] #["Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #["Torch_TARNet", "Torch_DragonNet", "Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #, "Torch_TLearner", "Torch_XLearner", "EconML_TLearner_Lasso", "Torch_ActionNet", "Torch_SLearner"]
                                #     cfg.plot_name_prefix = plots_folder+"v6_direct_vs_indirect"
                                #     plot_result(cfg, compare_axis_values=compare_axis_values)
                                # except Exception as e:
                                #     print("Error:", e)

                                # try:
                                #     cfg.model_names = ["Torch_DragonNet", "Torch_DragonNet_2", "Torch_DragonNet_4"] #["Torch_TARNet", "Torch_DragonNet", "Torch_CFRNet_0.01", "Torch_CFRNet_0.001", "Torch_CFRNet_0.0001"] #, "Torch_TLearner", "Torch_XLearner", "EconML_TLearner_Lasso", "Torch_ActionNet", "Torch_SLearner"]
                                #     cfg.plot_name_prefix = plots_folder+"v7_only_action_predictive"
                                #     plot_result(cfg, compare_axis_values=compare_axis_values)
                                # except Exception as e:
                                #     print("Error:", e)

    
if __name__ == "__main__":
    main()