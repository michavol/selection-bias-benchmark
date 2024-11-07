import argparse
import sys
import os
import wandb
from typing import Any
import pandas as pd 
import src.iterpretability.logger as log
# from src.iterpretability.experiments.experiments_ext import (
#     PredictiveSensitivity,
#     PropensitySensitivity,
#     NonLinearitySensitivity,
#     CohortSizeSensitivity,
# )
from src.iterpretability.simulators import TYSimulator, TSimulator

from src.iterpretability.experiments.sample_size_sensitivity import SampleSizeSensitivity
from src.iterpretability.experiments.important_feature_num_sensitivity import ImportantFeatureNumSensitivity
from src.iterpretability.experiments.propensity_scale_sensitivity import PropensityScaleSensitivity
from src.iterpretability.experiments.predictive_scale_sensitivity import PredictiveScaleSensitivity
from src.iterpretability.experiments.treatment_space_sensitivity import TreatmentSpaceSensitivity
from src.iterpretability.experiments.sample_size_sensitivity_visualization import SampleSizeSensitivityVisualization
from src.iterpretability.experiments.data_dimensionality_sensitivity import DataDimensionalitySensitivity
from src.iterpretability.experiments.feature_overlap_sensitivity import FeatureOverlapSensitivity
from src.iterpretability.experiments.expertise_sensitivity import ExpertiseSensitivity


from src.iterpretability.datasets.data_loader import load

# Hydra for configuration
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config_TY_tcga", version_base=None)
def main(cfg: DictConfig):
    ############################
    # 1. SETUP
    ############################
    # print(f"Working directory : {os.getcwd()}")
    # print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    
    # Set logging level
    log.add(sink=sys.stderr, level=cfg.log_level)

    ############################
    # 2. EXPERIMENTS
    ############################
    
    if cfg.experiment_name == "sample_size_sensitivity":
        exp = SampleSizeSensitivity(cfg)
        exp.run()

    elif cfg.experiment_name == "important_feature_num_sensitivity":
        exp = ImportantFeatureNumSensitivity(cfg)
        exp.run()

    elif cfg.experiment_name == "propensity_scale_sensitivity":
        exp = PropensityScaleSensitivity(cfg)
        exp.run()

    elif cfg.experiment_name == "predictive_scale_sensitivity":
        if cfg.simulator.simulation_type == "T":
            print("PredictiveScaleSensitivity is not supported for TSimulator")
            return None
        
        exp = PredictiveScaleSensitivity(cfg)
        exp.run()

    elif cfg.experiment_name == "treatment_space_sensitivity":
        if cfg.simulator.simulation_type == "T":
            print("PredictiveScaleSensitivity is not supported for TSimulator")
            return None
        
        exp = TreatmentSpaceSensitivity(cfg)
        exp.run()

    elif cfg.experiment_name == "sample_size_sensitivity_visualization":
        if cfg.simulator.dim_Y != 1 or cfg.simulator.num_T != 2:
            print("SampleSizeSensitivityVisualization is only supported for dim_Y=1 and dim_T=2")
            return None
        
        if cfg.simulator.simulation_type == "T":
            print("Make sure to only pick two treatments for this to work. And remove this if statement.")
            return None 
        
        exp = SampleSizeSensitivityVisualization(cfg)
        exp.run()

    elif cfg.experiment_name == "data_dimensionality_sensitivity":
        exp = DataDimensionalitySensitivity(cfg)
        exp.run()

    elif cfg.experiment_name == "feature_overlap_sensitivity":
        if cfg.simulator.simulation_type == "T":
            print("FeatureOverlapSensitivity is not supported for TSimulator")
            return None
        
        exp = FeatureOverlapSensitivity(cfg)
        exp.run()

    elif cfg.experiment_name == "expertise_sensitivity":
        exp = ExpertiseSensitivity(cfg)
        exp.run()
            
    else:
        raise ValueError(f"Invalid experiment name: {cfg.experiment_name}")
    
if __name__ == "__main__":
    main()