from pathlib import Path
import os
import catenets.models as cate_models
from catenets.models.diffpo import DiffPOLearner
import numpy as np
import pandas as pd
import wandb 
import shap
from PIL import Image
import src.iterpretability.logger as log
from sklearn.metrics import roc_auc_score
from src.plotting import (
    plot_results_datasets_compare, 
    plot_expertise_metrics,
    plot_performance_metrics,
    plot_performance_metrics_f_cf,
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
import time
# For contour plotting
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, LogisticRegressionCV, LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import auc

# Hydra for configuration
from omegaconf import DictConfig, OmegaConf

class ExperimentBase():
    """
    Base class for all experiments.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # Store configuration
        self.cfg = cfg

        # General experiment settings
        self.seeds = cfg.seeds
        self.n_splits = cfg.n_splits
        self.experiment_name = cfg.experiment_name
        self.simulation_type = cfg.simulator.simulation_type
        self.evaluate_inference = cfg.evaluate_inference

        # Load data
        if self.simulation_type == "TY":
            self.X,self.feature_names = load(cfg.dataset, 
                                            train_ratio=1, 
                                            debug=cfg.debug, 
                                            sim_type=self.simulation_type,
                                            directory_path_=cfg.directory_path+'/',
                                            repo_path = cfg.repo_path,
                                            n_samples=cfg.n_samples)
            self.outcomes = None
            self.discrete_outcome = cfg.simulator.num_binary_outcome == cfg.simulator.dim_Y # Currently the only discrete outcome is binary

        elif self.simulation_type == "T":
            self.X, self.outcomes, self.feature_names = load(cfg.dataset, 
                                                            train_ratio=1, 
                                                            debug=cfg.debug, 
                                                            directory_path_=cfg.directory_path+'/', 
                                                            repo_path = cfg.repo_path,
                                                            sim_type=self.simulation_type)
            
            # Check whether all entries in outcomes are integers
            self.discrete_outcome = cfg.simulator.num_binary_outcome
            
            cfg.simulator.num_T = self.outcomes.shape[1]

        else:
            raise ValueError(f"Simulation type {self.cfg.simulator.simulation_type} not supported.")

        # Initialize learners
        self.discrete_treatment = True # Currently simulation only supports discrete treatment

        # Results path and directories
        self.file_name = f"{self.cfg.results_dictionary_prefix}_{self.cfg.dataset}_T{self.cfg.simulator.num_T}_Y{self.cfg.simulator.dim_Y}_{self.cfg.simulator.simulation_type}sim_numbin{self.cfg.simulator.num_binary_outcome}"
        self.results_path = Path(self.cfg.results_path) / Path(self.cfg.experiment_name) / Path(self.file_name)

        if not self.results_path.exists():
            self.results_path.mkdir(parents=True, exist_ok=True)

        # Variables set by some or all experiments
        self.learners = None
        self.baseline_learners = None
        self.pred_learners = None
        self.prog_learner = None
        self.select_learner = None
        self.explanations = None
        self.all_important_features = None
        self.pred_features = None
        self.prog_features = None
        self.select_features = None
        self.true_num_swaps = None
        self.swap_counter = None
        self.training_times = {}

    def get_learners(self, 
                     num_features: int, 
                     seed: int = 123) -> dict:
        """
        Get learners for the experiment, based on the configuration settings.
        """
        if self.cfg.simulator.dim_Y > 1 and "Torch_XLearner" in self.cfg.model_names:
            raise ValueError("Torch_XLearner only supports one outcome dimension.")
        elif self.cfg.simulator.dim_Y > 1 and "EconML_SparseLinearDRLearner" in self.cfg.model_names:
            raise ValueError("EconML_SparseLinearDRLearner only supports one outcome dimension.")
        # Let user know that torch and diffpo do not support multiple treatments
        if self.cfg.simulator.num_T > 2 and ("Torch" in self.cfg.model_names or "DiffPOLearner" in self.cfg.model_names):
            raise ValueError("Torch and DiffPO models do not support multiple treatments. Only the first treatment will be used.")


        binary_y = self.discrete_outcome and self.cfg.simulator.num_binary_outcome == 1
        
        learners = {

                        "EconML_CausalForestDML": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                                model_name="EconML_CausalForestDML",
                                                                                discrete_treatment=self.discrete_treatment,
                                                                                discrete_outcome=self.discrete_outcome,
                                                                                seed=seed),

                        "EconML_DML": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                        model_name="EconML_DML",
                                                                        discrete_treatment=self.discrete_treatment,
                                                                        discrete_outcome=self.discrete_outcome,
                                                                        seed=seed),

                        "EconML_DRLearner": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                            model_name="EconML_DRLearner",
                                                                            discrete_treatment=self.discrete_treatment,
                                                                            discrete_outcome=self.discrete_outcome,
                                                                            seed=seed),

                        "EconML_DMLOrthoForest": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                                model_name="EconML_DMLOrthoForest",
                                                                                discrete_treatment=self.discrete_treatment,
                                                                                discrete_outcome=self.discrete_outcome,
                                                                                seed=seed),

                        "EconML_DROrthoForest": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                                model_name="EconML_DROrthoForest",
                                                                                discrete_treatment=self.discrete_treatment,
                                                                                discrete_outcome=self.discrete_outcome,
                                                                                seed=seed),

                        "EconML_ForestDRLearner": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                                    model_name="EconML_ForestDRLearner",
                                                                                    discrete_treatment=self.discrete_treatment,
                                                                                    discrete_outcome=self.discrete_outcome,
                                                                                    seed=seed),

                        "EconML_LinearDML": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                            model_name="EconML_LinearDML",
                                                                            discrete_treatment=self.discrete_treatment,
                                                                            discrete_outcome=self.discrete_outcome,
                                                                            seed=seed),

                        "EconML_LinearDRLearner": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                                model_name="EconML_LinearDRLearner",
                                                                                discrete_treatment=self.discrete_treatment,
                                                                                discrete_outcome=self.discrete_outcome,
                                                                                seed=seed),

                        "EconML_SparseLinearDML": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                                    model_name="EconML_SparseLinearDML",
                                                                                    discrete_treatment=self.discrete_treatment,
                                                                                    discrete_outcome=self.discrete_outcome,
                                                                                    seed=seed),


                        "EconML_SparseLinearDRLearner": cate_models.econml.EconMlEstimator2(self.cfg, 
                                                                                            model_name="EconML_SparseLinearDRLearner",
                                                                                            discrete_treatment=self.discrete_treatment,
                                                                                            discrete_outcome=self.discrete_outcome,
                                                                                            seed=seed), 
                        

                        "EconML_SLearner_Lasso": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                                    model_name="EconML_SLearner_Lasso",
                                                                                    discrete_treatment=self.discrete_treatment,
                                                                                    discrete_outcome=self.discrete_outcome,
                                                                                    seed=seed),

                        "EconML_TLearner_Lasso": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                                    model_name="EconML_TLearner_Lasso",
                                                                                    discrete_treatment=self.discrete_treatment,
                                                                                    discrete_outcome=self.discrete_outcome,
                                                                                    seed=seed),

                        "EconML_XLearner_Lasso": cate_models.econml.EconMlEstimator2(self.cfg,
                                                                                    model_name="EconML_XLearner_Lasso",
                                                                                    discrete_treatment=self.discrete_treatment,
                                                                                    discrete_outcome=self.discrete_outcome,
                                                                                    seed=seed),

                        "Torch_SLearner": cate_models.torch.SLearner(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_SLearner),

                        "Torch_TLearner": cate_models.torch.TLearner(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_TLearner),

                        "Torch_XLearner": cate_models.torch.XLearner(num_features, 
                                                                     binary_y=binary_y, 
                                                                     **self.cfg.Torch_XLearner),

                        "Torch_DRLearner": cate_models.torch.DRLearner(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_DRLearner),

                        "Torch_DragonNet": cate_models.torch.DragonNet(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_DragonNet),

                        "Torch_DragonNet_2": cate_models.torch.DragonNet(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_DragonNet_2),

                        "Torch_DragonNet_4": cate_models.torch.DragonNet(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_DragonNet_4),                                                                                            

                        "Torch_ActionNet": cate_models.torch.ActionNet(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_ActionNet),
                                                                        
                        # "Torch_FlexTENet": cate_models.torch.FlexTENet(num_features,
                        #                                                 binary_y=binary_y,
                        #                                                 **self.cfg.Torch_FlexTENet),

                        "Torch_PWLearner": cate_models.torch.PWLearner(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_PWLearner),

                        "Torch_RALearner": cate_models.torch.RALearner(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_RALearner),

                        "Torch_RLearner": cate_models.torch.RLearner(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_RLearner),

                        "Torch_TARNet": cate_models.torch.TARNet(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_TARNet),

                        "Torch_ULearner": cate_models.torch.ULearner(num_features,
                                                                        binary_y=binary_y,
                                                                        **self.cfg.Torch_ULearner),

                        "Torch_CFRNet_0.01": cate_models.torch.TARNet(num_features,
                                                                binary_y=binary_y,
                                                                **self.cfg.Torch_CRFNet_0_01),

                        "Torch_CFRNet_0.001": cate_models.torch.TARNet(num_features,
                                                                binary_y=binary_y,
                                                                **self.cfg.Torch_CRFNet_0_001),

                        "Torch_CFRNet_0.0001": cate_models.torch.TARNet(num_features,
                                                                binary_y=binary_y,
                                                                **self.cfg.Torch_CRFNet_0_0001),
                        "DiffPOLearner": DiffPOLearner(self.cfg,
                                                        num_features,
                                                        binary_y=binary_y,
                                                        ),

                    }

        # Deal with the case where the model is not available
        for name in self.cfg.model_names:
            if name not in learners:
                raise Exception(f"Unknown model name {name}.")
            
        # Only return learners from cfg.model_names
        return {k: v for k, v in learners.items() if k in self.cfg.model_names}
        
    def get_baseline_learners(self, 
                             seed: int = 123) -> dict:
        """
        Get baseline learners for the experiment, based on the configuration settings.
        These models will be used as a baseline for retrieving absolute outcomes from cate predictions. 
        """
        # Instantiate baseline learner for all possible treatment options
        baseline_learners = []
        
        if self.cfg.debug or self.cfg.dataset == "cytof_normalized_with_fastdrug" or self.cfg.dataset.startswith("melanoma"):
            cv = LeaveOneOut()
        else:
            cv = 5

        for t in range(self.cfg.simulator.num_T):
            if self.discrete_outcome == 1:
                base_model = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=cv, random_state=seed)
                #base_model = RandomForestClassifier(n_estimators=100, random_state=seed)
                model = MultiOutputClassifier(base_model)
                baseline_learners.append(model)
            else:
                base_model = LassoCV(cv=cv, n_alphas=5, max_iter=50, random_state=seed)
                #base_model = RandomForestRegressor(n_estimators=100, random_state=seed)
                model = MultiOutputRegressor(base_model)
                baseline_learners.append(model)

        return baseline_learners

    def get_select_learner(self,
                            seed: int = 123) -> dict:
        """
        Get learners for feature selection.
        """
        # Instantiate baseline learner for all possible treatment options
        if self.cfg.debug:
            cv = LeaveOneOut()
        else:
            cv = 5
        select_learner = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=cv, random_state=seed)
        #select_learner = RandomForestRegressor(n_estimators=100, max_depth=6)
        
        return select_learner
    
    def get_prog_learner(self,
                            seed: int = 123) -> dict:
        """
        Get learners for the prognostic part of the outcomes.
        """
        if self.cfg.debug:
            cv = LeaveOneOut()
        else:
            cv = 5
        base_model = LassoCV(cv=cv, n_alphas=5, max_iter=50, random_state=seed)

        #base_model = RandomForestRegressor(n_estimators=100, random_state=seed)
        prog_learner = MultiOutputRegressor(base_model)
        return prog_learner
    
    def get_pred_learners(self,
                            seed: int = 123) -> dict:
        """
        Get learners for the treatment-specific CATEs.
        """
        # Instantiate baseline learner for all possible treatment options
        pred_learners = []

        if self.cfg.debug or self.cfg.dataset.startswith("melanoma"):
            cv = LeaveOneOut()
        else:
            cv = 5
        #base_model = RandomForestRegressor(n_estimators=100, random_state=seed)
        for t in range(self.cfg.simulator.num_T):
            base_model = LassoCV(cv=cv, n_alphas=5, max_iter=50, random_state=seed)
            model = MultiOutputRegressor(base_model)
            pred_learners.append(model)

        return pred_learners
    
    def train_learners(self,
                        X_train: np.ndarray,
                        Y_train: np.ndarray,
                        T_train: np.ndarray,
                        outcomes_train: np.ndarray) -> None:
        """
        Train all learners.
        """
        self.training_times = {}
        for name in self.learners:
            # measure training time
            start_time = time.time()

            log.info(f"Fitting {name}.")
            if self.evaluate_inference:
                if not (name.startswith("EconML") or name.startswith("DiffPOLearner")):
                    raise ValueError("Only EconML models support inference.")
                
            if name == "DiffPOLearner":
                self.learners[name].train(X=X_train, y=Y_train, w=T_train, outcomes=outcomes_train) # fit before!!
            else:
                self.learners[name].train(X=X_train, y=Y_train, w=T_train) # fit before!!

            # measure training time
            end_time = time.time()
            self.training_times[name] = end_time - start_time


    def train_baseline_learners(self,
                                X_train: np.ndarray,
                                outcomes_train: np.ndarray,
                                T_train: np.ndarray) -> None:
        """
        Train all baseline learners.
        """
        for t in range(self.cfg.simulator.num_T):
            # Get all data points where treatment is t
            mask = T_train == t
            Y_train_t = outcomes_train[mask,t,:]
            X_train_t = X_train[mask,:]

            log.debug(
            f'Check baseline data for treatment {t}:'
            f'============================================'
            f'X_train: {X_train.shape}'
            f'\n{X_train}'
            f'\noutcomes_train: {outcomes_train.shape}'
            f'\n{outcomes_train}'
            f'\nT_train: {T_train.shape}'
            f'\n{T_train}'
            f'\nX_train_t: {X_train_t.shape}'
            f'\n{X_train_t}'
            f'\nY_train_t: {Y_train_t.shape}'
            f'\n{Y_train_t}'
            f'\n============================================\n\n'
            )
            # Check that there are data points for this treatment
            assert Y_train_t.shape[0] > 0, f"No data points for treatment {t}."

            log.info(f"Fitting baseline learner for treatment {t}.")

            if self.discrete_outcome:
                Y_train_t = Y_train_t.astype(bool)

            self.baseline_learners[t].fit(X_train_t, Y_train_t)

    def train_select_learner(self, 
                                X_train: np.ndarray,
                                T_train: np.ndarray) -> None:
        """
        Train the feature selection learner.
        """
        log.info(f"Fitting feature selection learner.")
        self.select_learner.fit(X_train, T_train)

    def train_prog_learner(self,
                            X_train: np.ndarray,
                            pred_outcomes_train: np.ndarray) -> None:
        """
        Train the model learning the prognostic part of the outcomes.
        Here the prognostic part is the average of all possible outcomes.
        """
        # pred_outcomes shape: n, num_T, dim_Y
        # X_train shape: n, dim_X

        # Compute the average of all possible outcomes
        prog_Y_train = np.mean(pred_outcomes_train, axis=1)

        # Train the model
        self.prog_learner.fit(X_train, prog_Y_train)

        log.debug(
            f'Check prog learner data:'
            f'============================================'
            f'X_train: {X_train.shape}'
            f'\n{X_train}'
            f'\npred_outcomes_train: {pred_outcomes_train.shape}'
            f'\n{pred_outcomes_train}'
            f'\nprog_Y_train: {prog_Y_train.shape}'
            f'\n{prog_Y_train}'
            f'\n============================================\n\n'
        )

    def train_pred_learner(self,
                            X_train: np.ndarray,
                            pred_cates_train: np.ndarray,
                            T_train: np.ndarray) -> None:
        """
        Train the model learning the CATEs.
        """
        for t in range(self.cfg.simulator.num_T):
            # Get treatment mask
            mask = T_train == t

            # For every patient with treatment t, we can simply compute the mean CATE
            # For others we need to add the cate for t, to get the desired cates (see biomarker attribution good notes)
            pred_Y_train = np.zeros((X_train.shape[0], self.cfg.simulator.dim_Y))
            pred_Y_train = pred_cates_train.mean(axis=1)
            pred_Y_train[~mask] -= pred_cates_train[~mask, t, :]
       
            # Train the model
            self.pred_learners[t].fit(X_train, pred_Y_train)

            log.debug(
                f'Check pred learner data for treatment {t}:'
                f'============================================'
                f'X_train: {X_train.shape}'
                f'\n{X_train[:10]}'
                f'\npred_cates_train: {pred_cates_train.shape}'
                f'\n{pred_cates_train[:10]}'
                f'\nT_train: {T_train.shape}'
                f'\n{T_train[:10]}'
                f'\npred_Y_train: {pred_Y_train.shape}'
                f'\n{pred_Y_train[:10]}'
                f'\n============================================\n\n'
            )

    def get_learner_explanations(self, 
                                 X: np.ndarray,
                                 return_explainer_names: bool = True,
                                 ignore_explainer_limit: bool = False,
                                 type: str = "pred") -> dict:
        """
        Get explanations for all learners.
        """
        learner_explainers = {}
        learner_explanations = {}
        explainer_names = {}

        for name in self.learners:
            log.info(f"Explaining {name}.")

            if ignore_explainer_limit:
                explainer_limit = X.shape[0]
            else:
                explainer_limit = self.cfg.explainer_limit

            if "EconML" in name:
                # EconML 
                if self.cfg.explainer_econml != "shap":
                    raise ValueError("Only shap is supported for EconML models.")

                if type == "pred":
                    cate_est = lambda X: self.learners[name].predict(X)
                    shap_values_avg = shap.Explainer(cate_est, X[:explainer_limit]).shap_values(X[:explainer_limit])
                    explainer_names[name] = "shap"
                    # shap_values = self.learners[name].explain(X[:explainer_limit], background_samples=None)

                    # treatment_names = self.learners[name].est.cate_treatment_names()
                    # output_names = self.learners[name].est.cate_output_names()

                    # # average absolute shaps over all treatment names
                    # shap_values_avg = np.zeros_like(shap_values[output_names[0]][treatment_names[0]].values)
                    # for output_name in output_names:
                    #     for treatment_name in treatment_names:
                    #         shap_values_avg += np.abs(shap_values[output_name][treatment_name].values)

                # Does not work yet!
                elif type == "prog":
                    if name == "EconML_SLearner_Lasso":
                        y0_est = lambda X: self.learners[name].est.overall_model.predict(np.hstack([X, np.ones((X.shape[0], 1)),np.zeros((X.shape[0], 1))]))
                        # pred outcomes shape should be: n, num_T, dim_Y
                    elif name == "EconML_TLearner_Lasso":
                        y0_est = lambda X: self.learners[name].est.models[0].predict(X)
                    elif name == "EconML_DML":
                        y0_est = lambda X: self.learners[name].predict_outcomes(X, outcomes=None)[:,0]
                    else:
                        raise ValueError(f"Model {name} not supported for prog learner explanations.")
                    
                    shap_values = shap.Explainer(y0_est, X[:explainer_limit]).shap_values(X[:explainer_limit])
                    shap_values_avg = shap_values
                    # average absolute shaps over all treatment names
                    # shap_values_avg = np.zeros_like(shap_values[0])
                    # for i in range(len(shap_values)):
                    #     shap_values_avg += np.abs(shap_values[i])
                
                
                learner_explanations[name] = shap_values_avg
                explainer_names[name] = "shap"
                
            else:
                # Also doesn't work yet!
                if type == "prog" and name in ["Torch_SLearner", "Torch_TLearner", "Torch_TARNet", "Torch_CFRNet_0.001", "Torch_CFRNet_0.01","Torch_CFRNet_0.0001","Torch_DragonNet","Torch_DragonNet_2","Torch_DragonNet_4"]:
                    # model_to_explain = self.learners[name]._po_estimators[0]
                    # X_to_explain = self._repr_estimator(X).squeeze()
                    y0_est = lambda X: self.learners[name].predict(X, return_po=True)[1]
                    learner_explanations[name] = shap.Explainer(y0_est, X[:explainer_limit]).shap_values(X[:explainer_limit])
                    explainer_names[name] = "shap"  

                elif type == "prog" and not name in ["Torch_SLearner", "Torch_TLearner", "Torch_TARNet", "Torch_CFRNet_0.001", "Torch_CFRNet_0.01","Torch_CFRNet_0.0001","Torch_DragonNet","Torch_DragonNet_2","Torch_DragonNet_4"]:
                    learner_explanations[name] = None
                    explainer_names[name] = None
                    #raise ValueError(f"Model {name} not supported for prog learner explanations.")
                
                else:
                    cate_est = lambda X: self.learners[name].predict(X)
                    learner_explanations[name] = shap.Explainer(cate_est, X[:explainer_limit]).shap_values(X[:explainer_limit])
                    explainer_names[name] = "shap"
                    # model_to_explain = self.learners[name]
                    # X_to_explain = X

                    # learner_explainers[name] = Explainer(
                    #     model_to_explain,
                    #     feature_names=list(range(X.shape[1])),
                    #     explainer_list=[self.cfg.explainer_torch],
                    # )
                    # learner_explanations[name] = learner_explainers[name].explain(
                    #     X[: explainer_limit]
                    # )
                    # learner_explanations[name] = learner_explanations[name][self.cfg.explainer_torch]
                    # explainer_names[name] = self.cfg.explainer_torch
                   


        # Check dimensions of explanations by looking at the shape of all explanation np arrays
        # for name in learner_explanations:
        #     log.debug(
        #         f"\nExplanations for {name} have shape {learner_explanations[name].shape}."
        #     )
            
        if return_explainer_names:
            return learner_explanations, explainer_names
        else:
            return learner_explanations
    
    def get_select_learner_explanations(self,
                                        X_reference: np.ndarray,
                                        X_to_explain: np.ndarray) -> np.ndarray:
        """
        Get explanations for the feature selection learner.
        Returns shape: n, dim_X, num_T.
        """
        explainer = shap.Explainer(self.select_learner, X_reference)
        shap_values = explainer(X_to_explain).values # Shape: n, dim_X, num_T

        return shap_values

    def get_prog_learner_explanations(self,
                                        X_reference: np.ndarray,
                                        X_to_explain: np.ndarray) -> np.ndarray:
        """
        Get explanations for the prognostic learner.
        Returns shape: n, dim_X, dim_Y.
        """
        # Get explanations for every outcome
        shap_values = np.zeros((X_to_explain.shape[0], X_to_explain.shape[1], self.cfg.simulator.dim_Y))
        for i in range(self.cfg.simulator.dim_Y):
            explainer = shap.Explainer(self.prog_learner.estimators_[i], X_reference, check_additivity=False)
            shap_values[:,:,i] = explainer(X_to_explain).values #, check_additivity=False for forest
       
        return shap_values
    
    def get_pred_learner_explanations(self,
                                        X_reference: np.ndarray,
                                        X_to_explain: np.ndarray) -> np.ndarray:
        """
        Get explanations for the treatment-specific CATEs (one vs. all).
        Returns shape: num_T, n, dim_X, dim_Y.
        """
        # Get explanations for every outcome and every reference treatment
        shap_values = np.zeros((self.cfg.simulator.num_T, X_to_explain.shape[0], X_to_explain.shape[1], self.cfg.simulator.dim_Y))
        shap_base_values = np.zeros((self.cfg.simulator.num_T, self.cfg.simulator.dim_Y))

        for t in range(self.cfg.simulator.num_T):
            for i in range(self.cfg.simulator.dim_Y):
                explainer = shap.Explainer(self.pred_learners[t].estimators_[i], X_reference)
                shap_explanation = explainer(X_to_explain)
                pred = self.pred_learners[t].predict(X_to_explain)
                shap_values[t,:,:,i] = explainer(X_to_explain).values #, check_additivity=False for forest
                shap_base_values[t,i] = shap_explanation.base_values[0]

        log.debug(
            f"\nExplanations for pred_learner have shape {shap_values.shape}."
            f"\n{shap_values[:,:10,:10,:]}"
        )

        return shap_values, shap_base_values
    

    def save_results(self, metrics_df: pd.DataFrame, compare_axis_values = None, plot_only: bool = False, save_df_only: bool = False, compare_axis=None, fig_type = "all") -> None:
        """
        Save results of the experiment.
        """
        file_name = self.file_name
        results_path = self.results_path

        if save_df_only:
            log.info(f"Saving intermediate results in {results_path}...")
            if not results_path.exists():
                results_path.mkdir(parents=True, exist_ok=True)
        else:
            log.info(f"Saving final results in {results_path}...")
            if not results_path.exists():
                results_path.mkdir(parents=True, exist_ok=True)

        if not plot_only:
            # Save metrics csv and metadata csv with the configuration
            if save_df_only:
                metrics_df.to_csv(
                    results_path / Path(file_name+"_checkpoint.csv")
                )
            else:
                metrics_df.to_csv(
                    results_path / Path(file_name+".csv")
                )
            OmegaConf.save(self.cfg, results_path / Path(file_name+".yaml"))

        # Choose figure size and legend position
        n_rows = len(self.cfg.metrics_to_plot)
        figsize = None #(10, 100)
        legend_position = 0.95
        log_x_axis = False

        # Save plots
        if self.cfg.plot_results and not save_df_only:
            if self.cfg.experiment_name == 'sample_size_sensitivity':
                compare_axis = "Propensity Scale"
                x_axis = "Sample Portion"
                x_label_name = r'$\omega_{\mathrm{ss}}$'
                x_values_to_plot = self.cfg.sample_sizes
            
            elif self.cfg.experiment_name == 'important_feature_num_sensitivity':
                compare_axis = "Feature Overlap"
                x_axis = "Num Important Features"
                x_label_name = r'$\omega_{\mathrm{IFs}}$'
                x_values_to_plot = self.cfg.important_feature_nums

            elif self.cfg.experiment_name == 'propensity_scale_sensitivity':
                # compare_axis = "Unbalancedness Exp"
                compare_axis = "Propensity Type"
                x_axis = "Propensity Scale"
                x_label_name = r'${\mathrm{\beta}}$'
                x_values_to_plot = self.cfg.propensity_scales
                log_x_axis = True

            elif self.cfg.experiment_name == 'predictive_scale_sensitivity':
                compare_axis = "Nonlinearity Scale"
                x_axis = "Predictive Scale"
                x_label_name = r'$\omega_{\mathrm{pds}}$'
                x_values_to_plot = self.cfg.predictive_scales

            elif self.cfg.experiment_name == 'treatment_space_sensitivity':
                compare_axis = "Binary Outcome"
                x_axis = "Treatment Options"
                x_label_name = r'$\omega_{\mathrm{Ts}}$'
                x_values_to_plot = self.cfg.num_Ts

            elif self.cfg.experiment_name == 'data_dimensionality_sensitivity':
                compare_axis = compare_axis
                x_axis = "Data Dimension"
                x_label_name = r'$Num Features$'
                x_values_to_plot = self.cfg.data_dims

            elif self.cfg.experiment_name == 'feature_overlap_sensitivity':
                compare_axis = "Overlap Type"
                x_axis = "Propensity Scale"
                x_label_name = r'$\omega_{\mathrm{pps}}$'
                x_values_to_plot = self.cfg.propensity_scales

            elif self.cfg.experiment_name == 'expertise_sensitivity':
                compare_axis = "Propensity Type"
                x_axis = "Alpha"
                x_label_name = r'$\alpha$'
                x_values_to_plot = self.cfg.alphas

            else:
                raise ValueError(f"Experiment {self.cfg.experiment_name} not supported for plotting.")

            if fig_type == "all":
                fig = plot_results_datasets_compare(
                            results_df=metrics_df,
                            model_names=self.cfg.model_names,
                            dataset=self.cfg.dataset,
                            compare_axis=compare_axis,
                            compare_axis_values=compare_axis_values,
                            x_axis=x_axis,
                            x_label_name=x_label_name,
                            x_values_to_plot=x_values_to_plot,
                            metrics_list=self.cfg.metrics_to_plot,
                            learners_list=self.cfg.model_names,
                            figsize=figsize,
                            legend_position=legend_position,
                            seeds_list=self.cfg.seeds,
                            n_splits=self.cfg.n_splits,
                            sharey="row",
                            legend_rows=1,
                            dim_X=self.X.shape[0],
                            log_x_axis=log_x_axis
                        )
                
            elif fig_type == "expertise":
                fig = plot_expertise_metrics(
                            results_df=metrics_df,
                            model_names=self.cfg.model_names,
                            dataset=self.cfg.dataset,
                            compare_axis=compare_axis,
                            compare_axis_values=compare_axis_values,
                            x_axis=x_axis,
                            x_label_name=x_label_name,
                            x_values_to_plot=x_values_to_plot,
                            metrics_list=self.cfg.metrics_to_plot,
                            learners_list=self.cfg.model_names,
                            figsize=figsize,
                            legend_position=legend_position,
                            seeds_list=self.cfg.seeds,
                            n_splits=self.cfg.n_splits,
                            sharey="row",
                            legend_rows=1,
                            dim_X=self.X.shape[0],
                            log_x_axis=log_x_axis
                        )
                
            elif fig_type == "performance":
                fig = plot_performance_metrics(
                            results_df=metrics_df,
                            model_names=self.cfg.model_names,
                            dataset=self.cfg.dataset,
                            compare_axis=compare_axis,
                            compare_axis_values=compare_axis_values,
                            x_axis=x_axis,
                            x_label_name=x_label_name,
                            x_values_to_plot=x_values_to_plot,
                            metrics_list=self.cfg.metrics_to_plot,
                            learners_list=self.cfg.model_names,
                            figsize=figsize,
                            legend_position=legend_position,
                            seeds_list=self.cfg.seeds,
                            n_splits=self.cfg.n_splits,
                            sharey="row",
                            legend_rows=1,
                            dim_X=self.X.shape[0],
                            log_x_axis=log_x_axis
                        )
                
            elif fig_type == "performance_f_cf":
                fig = plot_performance_metrics_f_cf(
                            results_df=metrics_df,
                            model_names=self.cfg.model_names,
                            dataset=self.cfg.dataset,
                            compare_axis=compare_axis,
                            compare_axis_values=compare_axis_values,
                            x_axis=x_axis,
                            x_label_name=x_label_name,
                            x_values_to_plot=x_values_to_plot,
                            metrics_list=self.cfg.metrics_to_plot,
                            learners_list=self.cfg.model_names,
                            figsize=figsize,
                            legend_position=legend_position,
                            seeds_list=self.cfg.seeds,
                            n_splits=self.cfg.n_splits,
                            sharey="row",
                            legend_rows=1,
                            dim_X=self.X.shape[0],
                            log_x_axis=log_x_axis
                        )
               

            # Save figure
            try:
                fig.savefig(results_path / f"{self.cfg.plot_name_prefix}.png", bbox_inches='tight')
            except:
                fig.savefig(results_path / f"{file_name}.png", bbox_inches='tight')

    def load_and_plot_results(self, compare_axis_values, fig_type: str = "all") -> None:
        """
        Load and plot results of the experiment.
        """
        file_name = self.file_name
        results_path = self.results_path

        log.info(f"Loading results from {results_path}...")
        if not results_path.exists():
            raise FileNotFoundError(f"Results path {results_path} does not exist.")

        # Load metrics
        metrics_df = pd.read_csv(results_path / Path(file_name+"_checkpoint.csv"))

        if self.cfg.experiment_name == "data_dimensionality_sensitivity":
            if self.cfg.compare_axis == "propensity":
                compare_axis = "Propensity Scale"
            elif self.cfg.compare_axis == "num_features":
                compare_axis = "Num Important Features"
            self.save_results(metrics_df, plot_only=True, compare_axis=compare_axis, compare_axis_values=compare_axis_values, fig_type=fig_type)
        else:
            self.save_results(metrics_df, plot_only=True, compare_axis_values=compare_axis_values, fig_type=fig_type)

    
    def compute_outcome_mse(self,
                            pred_outcomes: np.ndarray,
                            true_outcomes: np.ndarray,
                            T: np.ndarray = None) -> float:
        """
        Compute MSE for all outcomes.
        """
       
        # Only keep factual cates
        pred_outcomes_factual = np.zeros((pred_outcomes.shape[0], pred_outcomes.shape[2])) # n, dim_Y
        true_outcomes_factual = np.zeros((true_outcomes.shape[0], true_outcomes.shape[2]))
        pred_outcomes_factual_Y0 = np.zeros(((T==0).sum(), pred_outcomes.shape[2])) # n, dim_Y
        true_outcomes_factual_Y0 = np.zeros(((T==0).sum(), pred_outcomes.shape[2]))
        pred_outcomes_factual_Y1 = np.zeros(((T==1).sum(), pred_outcomes.shape[2])) # n, dim_Y
        true_outcomes_factual_Y1 = np.zeros(((T==1).sum(), pred_outcomes.shape[2]))

        pred_outcomes_cf = np.zeros((pred_outcomes.shape[0], pred_outcomes.shape[1]-1, pred_outcomes.shape[2])) # n, num_T-1, dim_Y
        true_outcomes_cf = np.zeros((true_outcomes.shape[0], true_outcomes.shape[1]-1, true_outcomes.shape[2]))
        pred_outcomes_cf_Y0 = np.zeros(((T==1).sum(), pred_outcomes.shape[1]-1, pred_outcomes.shape[2])) # n, dim_Y
        true_outcomes_cf_Y0 = np.zeros(((T==1).sum(), pred_outcomes.shape[1]-1, pred_outcomes.shape[2]))
        pred_outcomes_cf_Y1 = np.zeros(((T==0).sum(), pred_outcomes.shape[1]-1, pred_outcomes.shape[2])) # n, dim_Y
        true_outcomes_cf_Y1 = np.zeros(((T==0).sum(), pred_outcomes.shape[1]-1, pred_outcomes.shape[2]))

        counter_Y0 = 0
        counter_Y1 = 0
        for i in range(pred_outcomes.shape[0]):
            mask_factual = np.zeros(pred_outcomes.shape[1], dtype=bool)
            mask_cf = np.ones(pred_outcomes.shape[1], dtype=bool)
            mask_factual[T[i]] = True
            mask_cf[T[i]] = False

            pred_outcomes_factual[i,:] = pred_outcomes[i, mask_factual,:]
            true_outcomes_factual[i,:] = true_outcomes[i, mask_factual,:]
            pred_outcomes_cf[i,:,:] = pred_outcomes[i, mask_cf,:]
            true_outcomes_cf[i,:,:] = true_outcomes[i, mask_cf,:]

            if T[i] == 0:
                pred_outcomes_factual_Y0[counter_Y0,:] = pred_outcomes[i, mask_factual,:]
                true_outcomes_factual_Y0[counter_Y0,:] = true_outcomes[i, mask_factual,:]
                pred_outcomes_cf_Y1[counter_Y0,:,:] = pred_outcomes[i,mask_cf,:]
                true_outcomes_cf_Y1[counter_Y0,:,:] = true_outcomes[i,mask_cf,:]
                counter_Y0 += 1
            else:
                pred_outcomes_factual_Y1[counter_Y1,:] = pred_outcomes[i, mask_factual,:]
                true_outcomes_factual_Y1[counter_Y1,:] = true_outcomes[i, mask_factual,:]
                pred_outcomes_cf_Y0[counter_Y1,:,:] = pred_outcomes[i,mask_cf,:]
                true_outcomes_cf_Y0[counter_Y1,:,:] = true_outcomes[i,mask_cf,:]
                counter_Y1 += 1


        rmse_factual = mean_squared_error(true_outcomes_factual.reshape(-1), pred_outcomes_factual.reshape(-1), squared=False)
        rmse_factual_Y0 = mean_squared_error(true_outcomes_factual_Y0.reshape(-1), pred_outcomes_factual_Y0.reshape(-1), squared=False)
        rmse_factual_Y1 = mean_squared_error(true_outcomes_factual_Y1.reshape(-1), pred_outcomes_factual_Y1.reshape(-1), squared=False)
        rmse_cf = mean_squared_error(true_outcomes_cf.reshape(-1), pred_outcomes_cf.reshape(-1), squared=False)
        rmse_cf_Y0 = mean_squared_error(true_outcomes_cf_Y0.reshape(-1), pred_outcomes_cf_Y0.reshape(-1), squared=False)
        rmse_cf_Y1 = mean_squared_error(true_outcomes_cf_Y1.reshape(-1), pred_outcomes_cf_Y1.reshape(-1), squared=False)

        rmse_Y0 = mean_squared_error(np.concatenate([true_outcomes_factual_Y0.reshape(-1), true_outcomes_cf_Y0.reshape(-1)]), np.concatenate([pred_outcomes_factual_Y0.reshape(-1), pred_outcomes_cf_Y0.reshape(-1)]), squared=False)
        rmse_Y1 = mean_squared_error(np.concatenate([true_outcomes_factual_Y1.reshape(-1), true_outcomes_cf_Y1.reshape(-1)]), np.concatenate([pred_outcomes_factual_Y1.reshape(-1), pred_outcomes_cf_Y1.reshape(-1)]), squared=False)
        # Get variance of true outcomes, factual and cf
        # factual_std = np.std(true_outcomes_factual)
        # cf_std = np.std(true_outcomes_cf)
        factual_std = np.var(true_outcomes_factual)
        cf_std = np.var(true_outcomes_cf)

        log.debug(
            f"\nPred outcomes: {pred_outcomes.shape}: \n{pred_outcomes}"
            f"\n\nT: \n{T}"
            f"\n\nPred outcomes factual: {pred_outcomes_factual.shape}: \n{pred_outcomes_factual}"
        )
        
        return rmse_Y0, rmse_Y1, rmse_factual_Y0, rmse_factual_Y1, rmse_cf_Y0, rmse_cf_Y1, rmse_factual, rmse_cf, rmse_factual/factual_std, rmse_cf/cf_std, np.mean(true_outcomes_factual), np.mean(true_outcomes_cf), np.std(true_outcomes_factual), np.std(true_outcomes_cf)

    
    def compute_outcome_auroc(self,
                              pred_outcomes: np.ndarray,
                              true_outcomes: np.ndarray,
                              T: np.ndarray = None) -> float:
        """
        Compute AUROC for all outcomes.
        """
        # Only keep factual cates
        pred_outcomes_factual = np.zeros((pred_outcomes.shape[0], pred_outcomes.shape[2])) # n, dim_Y
        true_outcomes_factual = np.zeros((true_outcomes.shape[0], true_outcomes.shape[2]))
        pred_outcomes_cf = np.zeros((pred_outcomes.shape[0], pred_outcomes.shape[1]-1, pred_outcomes.shape[2])) # n, num_T-1, dim_Y
        true_outcomes_cf = np.zeros((true_outcomes.shape[0], pred_outcomes.shape[1]-1, true_outcomes.shape[2]))

        for i in range(pred_outcomes.shape[0]):
            mask_factual = np.zeros(pred_outcomes.shape[1], dtype=bool)
            mask_cf = np.ones(pred_outcomes.shape[1], dtype=bool)
            mask_factual[T[i]] = True
            mask_cf[T[i]] = False

            pred_outcomes_factual[i,:] = pred_outcomes[i, mask_factual,:]
            true_outcomes_factual[i,:] = true_outcomes[i, mask_factual,:]
            pred_outcomes_cf[i,:,:] = pred_outcomes[i, mask_cf,:]
            true_outcomes_cf[i,:,:] = true_outcomes[i, mask_cf,:]

        auroc_factual = roc_auc_score(true_outcomes_factual.reshape(-1), pred_outcomes_factual.reshape(-1))
        auroc_cf = roc_auc_score(true_outcomes_cf.reshape(-1), pred_outcomes_cf.reshape(-1))

        log.debug(
            f"\nPred outcomes: {pred_outcomes.shape}: \n{pred_outcomes}"
            f"\n\nT: \n{T}"
            f"\n\nPred outcomes factual: {pred_outcomes_factual.shape}: \n{pred_outcomes_factual}"
        )
        
        return auroc_factual, auroc_cf
    
        # if factual:
        #     # Only keep factual cates
        #     pred_outcomes_factual = np.zeros((pred_outcomes.shape[0], pred_outcomes.shape[2])) # dim_X, num_T, dim_Y
        #     true_outcomes_factual = np.zeros((true_outcomes.shape[0], true_outcomes.shape[2]))

        #     for i in range(pred_outcomes.shape[0]):
        #         mask = np.zeros(pred_outcomes.shape[1], dtype=bool)
        #         mask[T[i]] = True
        #         pred_outcomes_factual[i,:] = pred_outcomes[i, mask,:]
        #         true_outcomes_factual[i,:] = true_outcomes[i, mask,:]
        #     # f1 = f1_score(true_outcomes_factual.reshape(-1), pred_outcomes_factual.reshape(-1))
        #     f1 = roc_auc_score(true_outcomes_factual.reshape(-1), pred_outcomes_factual.clip(0,1).reshape(-1))
            
        # else:
        #     auroc = roc_auc_score(true_outcomes.reshape(-1), pred_outcomes.clip(0,1).reshape(-1))
        #     f1 = auroc
        #     #f1 = f1_score(true_outcomes.reshape(-1), pred_outcomes.reshape(-1))    
        
        # auroc = f1 #TODO: Change name to f1
        # return auroc


    def compute_overall_pehe(self,
                                pred_cates: np.ndarray,
                                true_cates: np.ndarray,
                                T: np.ndarray) -> float:
        """
        Compute average PEHE across all treatment options for all outcomes.
        """

        # Remove cates where basline and assigned treatment are the same - they evaluate to zero and are not informative
        if self.discrete_outcome:
            pred_cates = pred_cates.clip(-1,1)

        pehe_sum_total = 0
        pehe_normalized_sum_total = 0
        mean_sum_total = 0
        std_sum_total = 0

        for outcome_idx in range(pred_cates.shape[2]):
            pred_cates_curr = pred_cates[:,:,outcome_idx]
            true_cates_curr = true_cates[:,:,outcome_idx]

            counter = 0
            pehe_sum = 0
            pehe_normalized_sum = 0
            mean_sum = 0
            std_sum = 0

            for i in range(pred_cates.shape[1]):
                for j in range(i):
                    mask_j = T == j 
                    mask_i = T == i

                    n = np.sum(mask_j) + np.sum(mask_i)
                    counter += 1

                    pred_cates_curr_cate_j = pred_cates_curr[mask_j,i]
                    true_cates_curr_cate_j = true_cates_curr[mask_j,i]
                    # pred_cates_curr_cate_i = pred_cates_curr[mask_i,j]
                    # true_cates_curr_cate_i = true_cates_curr[mask_i,j]
                    pred_cates_curr_cate_i = -pred_cates_curr[mask_i,j] # Make sure to always record cate in same direction
                    true_cates_curr_cate_i = -true_cates_curr[mask_i,j] # TODO: Make sure this makes sense for multiple treatments & outcomes

                    pred_cates_curr_cate = np.concatenate((pred_cates_curr_cate_j, pred_cates_curr_cate_i)).reshape(-1)
                    true_cates_curr_cate = np.concatenate((true_cates_curr_cate_j, true_cates_curr_cate_i)).reshape(-1)

                    # Compute mean CATE
                    true_cates_mean = np.mean(true_cates_curr_cate)
                    mean_sum += true_cates_mean

                    # Compute std of CATE
                    true_cates_std = np.std(true_cates_curr_cate)
                    std_sum += true_cates_std

                    pehe_curr = mean_squared_error(true_cates_curr_cate, pred_cates_curr_cate)
                    pehe_sum += pehe_curr
                    pehe_normalized_sum += pehe_curr / np.var(true_cates_curr_cate)

                    log.debug(
                        f'Check pehe computation for outcome: {outcome_idx}, cate: {i}-{j}'
                        f'============================================'
                        f'\npred_cates: {pred_cates.shape}'
                        f'\n{pred_cates}'
                        f'\n\ntrue_cates: {true_cates.shape}'
                        f'\n{true_cates}'
                        f'\n\nT: {T.shape}'
                        f'\n{T}'
                        f'\n\npred_cates_curr: {pred_cates_curr.shape}'
                        f'\n{pred_cates_curr}'
                        f'\n\ntrue_cates_curr: {true_cates_mean.shape}'
                        f'\n{true_cates_curr}'
                        f'\n\nmask_i: {mask_i.shape}'
                        f'\n{mask_i}'
                        f'\n\nmask_j: {mask_j.shape}'
                        f'\n{mask_j}'
                        f'\n\pred_cates_curr_cate: {pred_cates_curr_cate.shape}'
                        f'\n{pred_cates_curr_cate}'
                        f'\n\ntrue_cates_curr_cate: {true_cates_curr_cate.shape}'
                        f'\n{true_cates_curr_cate}'
                        f'\n\ncounter is currently: {counter}'
                        f'\n\npehe_curr: {pehe_curr}'
                        f'\n\nmean_sum: {mean_sum}'
                        f'\n\nstd_sum: {std_sum}'
                        f'\n============================================\n\n'
                    )

            pehe = pehe_sum / counter
            pehe_normalized = pehe_normalized_sum / counter
            pehe_sum_total += pehe
            pehe_normalized_sum_total += pehe_normalized
            mean_sum_total += mean_sum / counter
            std_sum_total += std_sum / counter

        pehe_total = pehe_sum_total / pred_cates.shape[2]
        pehe_normalized_total = pehe_normalized_sum_total / pred_cates.shape[2]
        true_cates_mean_total = mean_sum_total / pred_cates.shape[2]
        true_cates_std_total = std_sum_total / pred_cates.shape[2]

        # pred_cates_filt = np.zeros((pred_cates.shape[0], pred_cates.shape[1]-1, pred_cates.shape[2])) # dim_X, num_T, dim_Y
        # true_cates_filt = np.zeros((true_cates.shape[0], true_cates.shape[1]-1, true_cates.shape[2]))

        # for i in range(pred_cates.shape[0]):
        #     mask = np.ones(pred_cates.shape[1], dtype=bool)
        #     mask[T[i]] = False

        #     pred_cates_filt[i,:,:] = pred_cates[i, mask,:]
        #     true_cates_filt[i,:,:] = true_cates[i, mask,:]

        # # Reshape to get all cates for each outcome in one dimension
        # pred_cates_filt = pred_cates_filt.reshape(-1)
        # true_cates_filt = true_cates_filt.reshape(-1)
        
        # # Compute PEHE 
        # pehe = mean_squared_error(true_cates_filt, pred_cates_filt)

        # Compute mean CATE
        # pred_cates_mean = np.mean(pred_cates_filt)
        # true_cates_mean = np.mean(true_cates_filt)

        # # Compute std of CATE
        # pred_cates_std = np.std(pred_cates_filt)
        # true_cates_std = np.std(true_cates_filt)

        # # Compute normalized PEHE
        # pehe_normalized = pehe / np.var(true_cates_filt)

        return pehe_total, pehe_normalized_total, true_cates_mean_total, true_cates_std_total

    def get_pred_cates(self, 
                       model_name: str, 
                       X: np.ndarray, 
                       T: np.ndarray,
                       outcomes_test: np.ndarray) -> np.ndarray:
        """
        Get the predicted CATEs for a model.
        """
        # Predict cate for every treatment option and use assigned treatment as baseline treatment.
        T0 = T
        T1 = np.zeros_like(T)
        pred_cates = np.zeros((X.shape[0], self.cfg.simulator.num_T, self.cfg.simulator.dim_Y))
        pred_cates_conf = np.zeros((X.shape[0], self.cfg.simulator.num_T, self.cfg.simulator.dim_Y, 2))

        for i in range(self.cfg.simulator.num_T):
            # Set to current treatment
            T1[:] = i

            # Deal with Torch models in case there are only two treatment options
            if self.cfg.simulator.num_T > 2:
                pred = self.learners[model_name].predict(X, T0=T0, T1=T1) # This predicts y[T1]-y[T0]

            elif model_name == "DiffPOLearner":
                mask = T1 == T0
                pred = self.learners[model_name].predict(X, T0=T0, T1=T1, outcomes=outcomes_test)
                pred[mask] = 0
                # if self.evaluate_inference:
                #     cates_conf = self.learners[model_name].est.effect_interval(X)
                if i == 0:
                    pred = -pred
              
            else:

                mask = T1 == T0
                
                pred = self.learners[model_name].predict(X) # This predicts y[1]-y[0]
                pred[mask] = 0
                # if self.evaluate_inference:
                #     cates_conf = self.learners[model_name].est.effect_interval(X)
                if i == 0:
                    pred = -pred
                    # cates_conf_lbs = -cates_conf[0]
                    # cates_conf_ups = -cates_conf[1]

            # print(cates_conf_lbs)
            # print(cates_conf_ups)

            pred = pred.reshape(-1, self.cfg.simulator.dim_Y)
            pred_cates[:, i, :] = pred.cpu().detach().numpy()

            # if self.evaluate_inference:
            #     cates_conf_lbs = cates_conf_lbs.reshape(-1, self.cfg.simulator.dim_Y)
            #     cates_conf_ups = cates_conf_ups.reshape(-1, self.cfg.simulator.dim_Y)
            #     pred_cates_conf[:, i, :, 0] = cates_conf_lbs
            #     pred_cates_conf[:, i, :, 1] = cates_conf_ups

            
        log.debug(f"Predicted CATEs for {model_name} have shape {pred_cates.shape}.")

        # if self.discrete_outcome:
        #     # Clip the values to the range [-1, 1]
        #     pred_cates = np.clip(pred_cates, -1, 1)

        #     # Round to the nearest integer
        #     pred_cates = np.rint(pred_cates)

        # Fill nan values with zeros
        if np.isnan(pred_cates).any():
            log.warning(f"There are nan values in the predicted CATEs for model: {model_name}. They were filled with 0s")
        pred_cates = np.nan_to_num(pred_cates)
        
        return pred_cates
    
    def get_pred_outcomes(self,
                            model_name: str,
                            pred_cates: np.ndarray,
                            X: np.ndarray,
                            T: np.ndarray) -> np.ndarray:
            """
            Get the predicted outcomes for a model.
            """
            # Get predicted outcomes for all treatment options
            pred_outcomes_baselines = np.zeros((X.shape[0], self.cfg.simulator.num_T, self.cfg.simulator.dim_Y))

            # Get directly predicted outcomes if available
            if model_name in ["DiffPOLearner", "EconML_SLearner_Lasso", "EconML_TLearner_Lasso", "EconML_DML"]:
                pred_outcomes = self.learners[model_name].predict_outcomes(X, T0=T, outcomes=None)
                return pred_outcomes
            
            elif model_name in ["Torch_ActionNet", "Torch_TARNet", "Torch_TLearner", "Torch_SLearner", "Torch_DragonNet_2", "Torch_DragonNet_4", "Torch_DragonNet", "TorchSNet", "Torch_CFRNet_0.001", "Torch_CFRNet_0.01", "Torch_CFRNet_0.0001"]:
                _, y0_pred, y1_pred = self.learners[model_name].predict(X, return_po=True)
                y0_pred = y0_pred.cpu().detach().numpy()
                y1_pred = y1_pred.cpu().detach().numpy()
                pred_outcomes = np.zeros((X.shape[0], self.cfg.simulator.num_T, self.cfg.simulator.dim_Y))
                pred_outcomes[:, 0, :] = y0_pred.reshape(-1, self.cfg.simulator.dim_Y)
                pred_outcomes[:, 1, :] = y1_pred.reshape(-1, self.cfg.simulator.dim_Y)
                return pred_outcomes

            else:
                for i in range(self.cfg.simulator.num_T):
                    # Get baseline outcomes
                    mask = T == i

                    # Fill in baseline outcomes according to selected treatments for each patient
                    if self.discrete_outcome:
                        baseline_preds = np.zeros((X[mask, :].shape[0], self.cfg.simulator.dim_Y))
                        baseline_preds_list = self.baseline_learners[i].predict_proba(X[mask, :])
                        for out_dim in range(self.cfg.simulator.dim_Y):
                            baseline_preds_curr = baseline_preds_list[out_dim][:,1]
                            baseline_preds[:,out_dim] = baseline_preds_curr

                    else:
                        baseline_preds = self.baseline_learners[i].predict(X[mask, :])
                        
                    baseline_preds = np.repeat(baseline_preds[:,np.newaxis,:], self.cfg.simulator.num_T, axis=1)

                    # Copy baseline predictions to all treatment options and make sure dimensions match
                    pred_outcomes_baselines[mask, :, :] = baseline_preds

                # Add predicted CATEs to baseline outcomes
                pred_outcomes = pred_outcomes_baselines + pred_cates

                log.debug(
                    f'Check predicted outcomes for model:'
                    f'============================================'
                    f'X: {X.shape}'
                    f'\n{X}'
                    f'\nT: {T.shape}'
                    f'\n{T}'
                    f'\npred_cates: {pred_cates.shape}'
                    f'\n{pred_cates}'
                    f'\npred_outcomes_baselines: {pred_outcomes_baselines.shape}'
                    f'\n{pred_outcomes_baselines}'
                    f'\npred_outcomes: {pred_outcomes.shape}'
                    f'\n{pred_outcomes}'
                    f'\n============================================\n\n'
                )

                # Clip to binary outcome
                # if self.discrete_outcome:
                #     pred_outcomes = np.clip(pred_outcomes, 0, 1)

                # Fill nan values with zeros and warn user
                if np.isnan(pred_outcomes).any():
                    log.warning("There are nan values in the predicted outcomes for. They were filled with zeros.")
                pred_outcomes = np.nan_to_num(pred_outcomes)
                return pred_outcomes
    
    def get_effect_cis(self,
                        model_name: str,
                        X: np.ndarray,
                        T: np.ndarray) -> np.ndarray:
        """
        Get the confidence intervals for the predicted CATEs.
        """
        # Predict cate for every treatment option and use assigned treatment as baseline treatment.
        T0 = T
        T1 = np.zeros_like(T)
        pred_cates_conf = np.zeros((2, X.shape[0], self.cfg.simulator.num_T, self.cfg.simulator.dim_Y))

        for i in range(self.cfg.simulator.num_T):
            # Set to current treatment
            T1[:] = i

            effect_cis = self.learners[model_name].infer_effect_ci(X=X, T0=T0) # dim: 2, n, dim_Y (2 for lower and upper bound of confidence interval)
            pred_cates_conf[:, :, i, :] = effect_cis

        return pred_cates_conf


    def get_swap_statistics_single_outcome(self,
                                           T: np.ndarray, 
                                           outcomes: np.ndarray, # n, num_T
                                           pred_cates: np.ndarray, # n, num_T
                                           shap_values_pred: np.ndarray, # num_T, n, dim_X
                                           shap_base_values_pred: np.ndarray, # num_T
                                           k: int = 1, 
                                           threshold: int = 0):
        """
        Evaluates for a given decisions threshold and number of decision variables k,  
        how well the personalized shap values predict whether the treatment should be swapped.
        """
        tp, fp, tn, fn = 0,0,0,0 # slightly biased, but avoids division by 0

        for i in range(T.shape[0]):
            # Check whether the true outcomes would speak for swapping the treatment
            outcome = outcomes[i,T[i]]
            outcome_mean = np.mean(outcomes[i])
            swap_true = outcome < outcome_mean # swap if below mean
            

            # print(f"Outcome: {outcome}, Outcome mean: {outcome_mean}, Swap true: {swap_true}")
            # print(f"Pred cates: {pred_cates[i,:]}")
            # Check whether the SHAP values would speak for swapping the treatment
            # get average cate
            swap_pred = np.mean(pred_cates[i,:]) > threshold
            

            ## OLD: for top k evaluation
            # shap_values = shap_values_pred[T[i],i,:]
            # Get the top k features in terms of absolute SHAP values
            # top_k = np.argsort(np.abs(shap_values))[-k:] #[-k] to check for one feature
            # swap_pred = np.sum(shap_values[top_k]) + shap_base_values_pred[T[i]] > threshold

            if swap_true and swap_pred:
                tp += 1
            elif swap_true and not swap_pred:
                fn += 1
            elif not swap_true and swap_pred:
                fp += 1
            else:
                tn += 1

        # Compute the FPR and TPR and precision and recall
        fpr = fp / (fp + tn) if fp + tn > 0 else np.nan
        tpr = tp / (tp + fn) if tp + fn > 0 else np.nan
        precision = tp / (tp + fp) if tp + fp > 0 else np.nan
        recall = tp / (tp + fn) if tp + fn > 0 else np.nan

        return fpr, tpr, precision, recall
    
    
    def compute_swap_metrics_single_outcome(self,
                                            T: np.ndarray,
                                            outcomes: np.ndarray, # n, num_T
                                            pred_cates: np.ndarray, # n, num_T
                                            shap_values_pred: np.ndarray, # num_T, n, dim_X
                                            shap_base_values_pred: np.ndarray, # num_T
                                            k: int = 1,) -> dict:
        """
        Computes the swap metrics for all outcomes.
        """
        # Get decision thresholds for auroc and auprc computation
        thresholds = []
        thresholds = list(pred_cates.reshape(-1))
        
        ## For top k auroc
        # for i in range(T.shape[0]):
        #     shap_values = shap_values_pred[T[i],i,:]

        #     # Get the top k features in terms of absolute SHAP values
        #     top_k = np.argsort(np.abs(shap_values))[-k:]
        #     thresholds.append(np.sum(shap_values[top_k]) + shap_base_values_pred[T[i]])
        
        # Only use unique thresholds
        thresholds = np.unique(thresholds)

        if thresholds.shape[0] == 1:
            thresholds = np.array([-1, thresholds[0],1])

        # Only leave 200 thresholds to accelerate computation of scores
        if len(thresholds) >= 200:
            step = len(thresholds)//200
        else:
            step = 1
        thresholds = thresholds[::step]

        # Iterate through all thresholds and compute statistics
        fprs, tprs, precisions, recalls = [], [], [], []
        for threshold in thresholds:
            fpr, tpr, precision, recall = self.get_swap_statistics_single_outcome(T, 
                                                                                  outcomes, 
                                                                                  pred_cates,
                                                                                  shap_values_pred, 
                                                                                  shap_base_values_pred,
                                                                                  k, threshold)

            fprs.append(fpr)
            tprs.append(tpr)

            if precision is not np.nan:
                precisions.append(precision)
            # else:
            #     log.info("Precision is nan")
                
            if recall is not np.nan:
                recalls.append(recall)
            # else:
            #     log.info("Recall is nan")

        # Compute auroc and auprc
        # Sort fprs and tprs by fpr
        fprs_sort, tprs_sort = zip(*sorted(zip(fprs, tprs)))
        fprs_sort, tprs_sort = np.array(fprs_sort), np.array(tprs_sort)

        # Compute the AUC
        roc_auc = auc(fprs_sort, tprs_sort)

        # Compute the AUC for the precision-recall curve
        if len(recalls) > 1 and len(precisions) > 1:
            # Sort precisions and recalls by recall
            recalls_sort, precisions_sort = zip(*sorted(zip(recalls, precisions)))
            recalls_sort, precisions_sort = np.array(recalls_sort), np.array(precisions_sort)
        
            pr_auc = auc(recalls_sort, precisions_sort)
        else:
            pr_auc = -1

        return roc_auc, pr_auc
    
    def compute_swap_metrics(self,
                            T: np.ndarray,
                            true_outcomes: np.ndarray,
                            pred_cates: np.ndarray,
                            shap_values_pred: np.ndarray,
                            shap_base_values_pred: np.ndarray,
                            k: int = 1) -> dict:
        """
        Compute swap metrics for all outcomes.
        """
        roc_aucs, pr_aucs = [], []

        for i in range(self.cfg.simulator.dim_Y):
            roc_auc, pr_auc = self.compute_swap_metrics_single_outcome(T, 
                                                                      true_outcomes[:,:,i], 
                                                                      pred_cates[:,:,i],
                                                                      shap_values_pred[:,:,:,i], 
                                                                      shap_base_values_pred[:,i],
                                                                      k)
            roc_aucs.append(roc_auc)
            pr_aucs.append(pr_auc)
           
        roc_auc_total = np.mean(roc_aucs)
        pr_auc_total = np.mean(pr_aucs)

        # Compute swap percentage with threshold 0
        counter = 0
        swap_counter = 0
        policy_precision = 0
        predicted_precision = 0
        for j in range(self.cfg.simulator.dim_Y):
            for i in range(T.shape[0]):
                # Check whether the true outcomes would speak for swapping the treatment
                true_outcome = true_outcomes[i,T[i],j]
                true_outcome_mean = np.mean(true_outcomes[i,:,j])
                swap_true = true_outcome < true_outcome_mean # swap if below mean
                counter += 1.0
                swap_counter += swap_true
  
        return roc_auc_total, pr_auc_total, swap_counter/counter
    
    def compute_ci_coverage(self,
                            pred_effect_cis: np.ndarray, # dim: 2, n, num_T-1, dim_Y
                            true_cates: np.ndarray, # n, num_T, dim_Y
                            T: np.ndarray) -> float:
        """
        Compute the coverage of the confidence intervals.
        """

        ci_coverage = 0
        counter = 0
        for i in range(pred_effect_cis.shape[2]):
            for j in range(pred_effect_cis.shape[3]):
                lb = pred_effect_cis[0,:,i,j]
                ub = pred_effect_cis[1,:,i,j]
                true_cate = true_cates[:,i,j]

                # Only consider the cates where the baseline and assigned treatment are different
                mask = T != i
                lb = lb[mask]
                ub = ub[mask]
                true_cate = true_cate[mask]

                ci_coverage += np.sum((lb <= true_cate) & (true_cate <= ub))
                counter += len(true_cate)

                log.debug(
                    f'Check ci coverage computation for outcome: {j}, cate: {i}'
                    f'============================================'
                    f'\ntrue_cates: {true_cate.shape}'
                    f'\n{true_cate}'
                    f'\n\nT: {T.shape}'
                    f'\n{T}'
                    f'\n\nlb: {lb.shape}'
                    f'\n{lb}'
                    f'\n\nub: {ub.shape}'
                    f'\n{ub}'
                    f'\n\ncoverage is currently: {ci_coverage}'
                    f'\n============================================\n\n'
                )

        ci_coverage /= counter
        
        return ci_coverage

    def compute_expertise_metrics(self,
                                    T: np.ndarray,
                                    outcomes: np.ndarray,
                                    type: str = "prognostic") -> tuple:
        """
        Compute the prognostic or treatment expertise.
        """
        if self.cfg.simulator.num_T != 2:
            raise ValueError("Expertise metrics can only be computed for binary treatments.")

        # Get potential outcomes
        y0 = outcomes[:, 0, 0]
        y1 = outcomes[:, 1, 0]

        # prognostic expertise calculation
        if type == "predictive" or type == "prognostic" or type == "treatment":
            if type == "predictive":
                _, gt_bins = np.histogram(y1 - y0, bins="auto")
                gt_uncond_result = y1 - y0

            elif type == "prognostic":
                _, gt_bins = np.histogram(y0, bins="auto")
                gt_uncond_result = y0

            elif type == "treatment":
                _, gt_bins = np.histogram(y1, bins="auto")
                gt_uncond_result = y1

            else:
                raise ValueError("Invalid expertise type. Choose between 'predictive', 'prognostic' or 'treatment'.")

            actions, _ = np.histogram(T, bins='auto')
            actions = actions / T.shape[0]
            actentropy = -np.sum(actions * np.ma.log(actions))

            num_ones = np.sum(T) / T.shape[0]

            # cond = -np.mean(propensity_test * np.log(propensity_test) + (1 - propensity_test) * np.log(1 - propensity_test))
            gt_uncond_hist, gt_uncond_bins = np.histogram(gt_uncond_result, bins=gt_bins)
            gt_uncond_hist = gt_uncond_hist / gt_uncond_result.shape[0]
            gt_uncond_hist = gt_uncond_hist[gt_uncond_hist != 0]
            gt_uncond_entropy = -np.sum(gt_uncond_hist * np.log(gt_uncond_hist))

            gt_cond_one = gt_uncond_result * T
            gt_cond_one = gt_cond_one[gt_cond_one != 0]
            gt_one_hist, _ = np.histogram(gt_cond_one, bins=gt_bins)
            gt_one_hist = gt_one_hist / gt_cond_one.shape[0]
            gt_one_hist = gt_one_hist[gt_one_hist != 0]
            gt_one_entropy = -np.sum(gt_one_hist * np.log(gt_one_hist))

            gt_cond_zero = gt_uncond_result * (1 - T)
            gt_cond_zero = gt_cond_zero[gt_cond_zero != 0]
            gt_zero_hist, _ = np.histogram(gt_cond_zero, bins=gt_bins)
            gt_zero_hist = gt_zero_hist / gt_cond_zero.shape[0]
            gt_zero_hist = gt_zero_hist[gt_zero_hist != 0]
            gt_zero_entropy = -np.sum(gt_zero_hist * np.log(gt_zero_hist))

            gt_expertise = gt_uncond_entropy - num_ones * gt_one_entropy - (1 - num_ones) * gt_zero_entropy
            gt_expertise = gt_expertise / actentropy

        elif type == "total":
            _, gt1_bins, gt0_bins = np.histogram2d(y1, y0)

            actions, _ = np.histogram(T, bins='auto')
            actions = actions / T.shape[0]
            actentropy = -np.sum(actions * np.ma.log(actions))

            num_ones = np.sum(T) / T.shape[0]

            # cond = -np.mean(propensity_test * np.log(propensity_test) + (1 - propensity_test) * np.log(1 - propensity_test))

            gt_uncond_hist, _, _ = np.histogram2d(y1, y0, bins=[gt1_bins, gt0_bins])
            gt_uncond_hist = gt_uncond_hist / y1.shape[0] #/ y0.shape[0]
            gt_uncond_hist = gt_uncond_hist[gt_uncond_hist != 0]
            gt_uncond_entropy = -np.sum(gt_uncond_hist * np.log(gt_uncond_hist))

            gt_cond_one1 = y1 * T
            gt_cond_one1 = gt_cond_one1[gt_cond_one1 != 0]
            gt_cond_one0 = y0 * T
            gt_cond_one0 = gt_cond_one0[gt_cond_one0 != 0]
            gt_one_hist, _, _ = np.histogram2d(gt_cond_one1, gt_cond_one0, bins=[gt1_bins, gt0_bins])
            gt_one_hist = gt_one_hist / gt_cond_one1.shape[0] #/ gt_cond_one0.shape[0]
            gt_one_hist = gt_one_hist[gt_one_hist != 0]
            gt_one_entropy = -np.sum(gt_one_hist * np.log(gt_one_hist))

            gt_cond_zero1 = y1 * (1 - T)
            gt_cond_zero1 = gt_cond_zero1[gt_cond_zero1 != 0]
            gt_cond_zero0 = y0 * (1 - T)
            gt_cond_zero0 = gt_cond_zero0[gt_cond_zero0 != 0]
            gt_zero_hist, _, _ = np.histogram2d(gt_cond_zero1, gt_cond_zero0, bins=[gt1_bins, gt0_bins])
            gt_zero_hist = gt_zero_hist / gt_cond_zero1.shape[0] #/ gt_cond_zero0.shape[0]
            gt_zero_hist = gt_zero_hist[gt_zero_hist != 0]
            gt_zero_entropy = -np.sum(gt_zero_hist * np.log(gt_zero_hist))

            gt_expertise = gt_uncond_entropy - num_ones * gt_one_entropy - (1 - num_ones) * gt_zero_entropy
            gt_expertise = gt_expertise / actentropy
        
        else:
            raise ValueError("Invalid expertise type. Choose between 'predictive', 'prognostic' or 'treatment'.")
        
        return gt_expertise
    
    def compute_incontext_variability(self,
                                      T: np.ndarray,
                                        propensities: np.ndarray) -> float:
        """
        Compute the in-context variability.
        """
        # Sum over contribution of all patients
        props = propensities.reshape(-1)
        props = props[props != 0]
        cond_entropy = 2*-np.mean(props * np.log(props))

        # Get action entropy
        actions, _ = np.histogram(T, bins='auto')
        actions = actions / T.shape[0]
        actentropy = -np.sum(actions * np.ma.log(actions))

        return cond_entropy / actentropy

    def get_pred_assignment_precision(self, pred_outcomes, true_outcomes):
        # Compute swap percentage with threshold 0
        counter = 0
        correct_counter = 0
        for j in range(self.cfg.simulator.dim_Y):
            for i in range(pred_outcomes.shape[0]):
                # Check whether the true outcomes would speak for swapping the treatment
                true_y0 = true_outcomes[i,0,j]
                true_y1 = true_outcomes[i,1,j]
                pred_y0 = pred_outcomes[i,0,j]
                pred_y1 = pred_outcomes[i,1,j]

                true_ranking = true_y0 < true_y1 
                pred_ranking = pred_y0 < pred_y1 

                if (true_ranking == 0 and pred_ranking == 0) or ((true_ranking == 1 and pred_ranking == 1)):
                    correct_counter += 1
                
                counter += 1.0
        return correct_counter / counter

    def compute_metrics(self,
                        results_data: dict,
                        sim: SimulatorBase,
                        X_train: np.ndarray,
                        Y_train: np.ndarray,
                        T_train: np.ndarray,
                        X_test: np.ndarray,
                        Y_test: np.ndarray,
                        T_test: np.ndarray,
                        outcomes_train: np.ndarray,
                        outcomes_test: np.ndarray,
                        propensities_train: np.ndarray,
                        propensities_test: np.ndarray,
                        x_axis_value: float,
                        x_axis_name: str,
                        compare_value: float,
                        compare_name: str,
                        seed: int,
                        split_id: int) -> dict:
        """
        Compute metrics for a given experiment.
        """
        # Get learners
        self.baseline_learners = self.get_baseline_learners(seed=seed)
        self.learners = self.get_learners(num_features=X_train.shape[1], seed=seed)

        # Train learners
        if self.cfg.train_baseline_learner:
            self.train_baseline_learners(X_train, outcomes_train, T_train)
        self.train_learners(X_train, Y_train, T_train, outcomes_train)

        # Get treatment distribution for train and test
        train_treatment_distribution = np.bincount(T_train) / len(T_train) 
        test_treatment_distribution = np.bincount(T_test) / len(T_test)

        if self.cfg.simulator.num_T == 2: # Such that it can be plotted
            train_treatment_distribution = train_treatment_distribution[0]
            test_treatment_distribution = test_treatment_distribution[0]

        # Get learner explanations
        (learner_explanations, _) = self.get_learner_explanations(X=X_test, type="pred") if self.cfg.evaluate_explanations else (None, None)
        (learner_prog_explanations, _) = self.get_learner_explanations(X=X_test, type="prog") if self.cfg.evaluate_prog_explanations else (None, None)

        # Get and train select learner - treatment selection model
        if self.cfg.evaluate_in_context_variability:
            self.select_learner = self.get_select_learner(seed=seed)
            self.train_select_learner(X_train, T_train)
            propensities_pred_test = self.select_learner.predict_proba(X_test)
            propensities_pred_train = self.select_learner.predict_proba(X_train)

        # # get auroc for propensity score
        # propensity_auc = roc_auc_score(T_test, prop_test[:,1])
        # # propensity_auc = auc(fpr, tpr)
        # print(propensity_auc)
        # quit()
        # select_learner_explanations = self.get_select_learner_explanations(X_reference=X_train, 
        #                                                                     X_to_explain=X_test)

        # Compute metrics
        baseline_metrics = {} # First model in list will be used as baseline

        for i, model_name in enumerate(self.cfg.model_names):
            # try:
            # Compute attribution accuracies
            # print(f"Computing attribution accuracies for model: {model_name}...")
            # print(learner_explanations)
            # print(f"Attribution explanations: {learner_explanations[model_name].shape}")
            if self.cfg.evaluate_explanations and learner_explanations[model_name] is not None:
                attribution_est = np.abs(learner_explanations[model_name]) 
                pred_acc_scores_all_features = attribution_accuracy(self.all_important_features, attribution_est) 
                pred_acc_scores_predictive_features = attribution_accuracy(self.pred_features, attribution_est) 
                pred_acc_scores_prog_features = attribution_accuracy(self.prog_features, attribution_est) 
                pred_acc_scores_selective_features = attribution_accuracy(self.select_features, attribution_est) 

            else:
                pred_acc_scores_all_features = -1
                pred_acc_scores_predictive_features = -1
                pred_acc_scores_prog_features = -1
                pred_acc_scores_selective_features = -1

            if self.cfg.evaluate_prog_explanations and learner_prog_explanations[model_name] is not None:
                prog_attribution_est = np.abs(learner_prog_explanations[model_name]) 
                prog_acc_scores_all_features = attribution_accuracy(self.all_important_features, prog_attribution_est) 
                prog_acc_scores_predictive_features = attribution_accuracy(self.pred_features, prog_attribution_est) 
                prog_acc_scores_prog_features = attribution_accuracy(self.prog_features, prog_attribution_est) 
                prog_acc_scores_selective_features = attribution_accuracy(self.select_features, prog_attribution_est) 

            else:
                prog_acc_scores_all_features = -1
                prog_acc_scores_predictive_features = -1
                prog_acc_scores_prog_features = -1
                prog_acc_scores_selective_features = -1

            # Compute predicted cates and outcomes
            pred_cates = self.get_pred_cates(model_name, X_test, T_test, outcomes_test)
            pred_cates_train = self.get_pred_cates(model_name, X_train, T_train, outcomes_train)

            pred_outcomes = self.get_pred_outcomes(model_name, pred_cates, X_test, T_test)
            pred_outcomes_train = self.get_pred_outcomes(model_name, pred_cates_train, X_train, T_train)

            pred_effect_cis = self.get_effect_cis(model_name, X_test, T_test) if self.evaluate_inference else None

     
            # quit()
            # pred_cates_train = self.get_pred_cates(model_name, X_train, T_train, outcomes_train)
            # pred_outcomes_train = self.get_pred_outcomes(pred_cates_train, X_train, T_train)


            # Train models for treat-specific explanations
            temp = self.discrete_outcome
            self.discrete_outcome = 0
            self.pred_learners = self.get_pred_learners(seed=seed) # One for each treatment as reference treatment
            self.prog_learner = self.get_prog_learner(seed=seed) # One for the average over all treatments
            self.discrete_outcome = temp

            # self.train_prog_learner(X_train, pred_outcomes_train)
            # self.train_pred_learner(X_train, pred_cates_train, T_train)
            # Get treat-specific explanations
            # prog_learner_explanations = self.get_prog_learner_explanations(X_reference=X_train,
            #                                                                 X_to_explain=X_test)
            # if self.cfg.evaluate_explanations:
            #     pred_learner_explanations, pred_learner_base_values = self.get_pred_learner_explanations(X_reference=X_train,
            #                                                                                         X_to_explain=X_test)
            # else: 
            #     pred_learner_explanations = np.zeros((self.cfg.simulator.num_T, X_test.shape[0], X_test.shape[1], self.cfg.simulator.dim_Y))
            #     pred_learner_base_values = np.zeros((self.cfg.simulator.num_T, self.cfg.simulator.dim_Y))

            ## COMPUTE METRICS ##
            # Compute PEHE
            true_cates = sim.get_true_cates(X_test, T_test, outcomes_test)

            # print(f"Pred cates: {pred_cates[:5]}")
            # print(f"Pred outcomes: {pred_outcomes[:5]}")
            # print(f"True cates: {true_cates[:5]}")
            # print(f"True outcomes: {outcomes_test[:5]}")
            # quit()
            (
                pehe, 
                pehe_normalized, 
                true_cates_mean, 
                true_cates_std
            ) = self.compute_overall_pehe(pred_cates, true_cates, T_test)

            (
                rmse_Y0, rmse_Y1, rmse_factual_Y0, rmse_factual_Y1, rmse_cf_Y0, rmse_cf_Y1, 
                factual_outcomes_rmse, cf_outcomes_rmse,
                factual_outcomes_rmse_normalized, cf_outcomes_rmse_normalized,
                factual_outcomes_mean, cf_outcomes_mean,
                factual_outcomes_std, cf_outcomes_std
            )= self.compute_outcome_mse(pred_outcomes, outcomes_test, T_test)
            f_cf_diff = np.abs(factual_outcomes_rmse - cf_outcomes_rmse)
            f_cf_diff_norm = np.abs(factual_outcomes_rmse_normalized - cf_outcomes_rmse_normalized)


            # Get aurocs in case of discrete outcomes
            cf_outcomes_auroc = -1
            factual_outcomes_auroc = -1
            if self.discrete_outcome:
                factual_outcomes_auroc, cf_outcomes_auroc = self.compute_outcome_auroc(pred_outcomes, outcomes_test, T_test)

            # Compute personalized explanations metrics
            #k_tre = self.cfg.simulator.num_pred_features * self.cfg.simulator.num_T 
            k_all = X_test.shape[1]

            # Compute confidence intervall coverage
            ci_coverage = self.compute_ci_coverage(pred_effect_cis, true_cates, T_test) if self.evaluate_inference else -1

            # Compute expertise metrics
            # Get learned policy
            T_pred = pred_outcomes[:, :, 0].argmax(axis=1)

            try:
                gt_pred_expertise = self.compute_expertise_metrics(T_train, outcomes_train, type="predictive")
                gt_prog_expertise = self.compute_expertise_metrics(T_train, outcomes_train, type="prognostic")  
                gt_tre_expertise = self.compute_expertise_metrics(T_train, outcomes_train, type="treatment") 
            
                updated_gt_pred_expertise = self.compute_expertise_metrics(T_pred, outcomes_test, type="predictive")
                updated_gt_prog_expertise = self.compute_expertise_metrics(T_pred, outcomes_test, type="prognostic")  
                updated_gt_tre_expertise = self.compute_expertise_metrics(T_pred, outcomes_test, type="treatment") 

                gt_total_expertise = self.compute_expertise_metrics(T_train, outcomes_train, type="total") if self.discrete_outcome == 0 else -1
                es_pred_expertise = self.compute_expertise_metrics(T_train, pred_outcomes_train, type="predictive")
                es_prog_expertise = self.compute_expertise_metrics(T_train, pred_outcomes_train, type="prognostic")
                es_tre_expertise = self.compute_expertise_metrics(T_train, pred_outcomes_train, type="treatment")
                es_total_expertise = self.compute_expertise_metrics(T_train, pred_outcomes_train, type="total") if self.discrete_outcome == 0 else -1

            except:
                print("Error while computing expertise")
                gt_pred_expertise = -1
                gt_prog_expertise = -1  
                gt_tre_expertise = -1
                gt_total_expertise = -1
                updated_gt_pred_expertise = -1
                updated_gt_prog_expertise = -1
                updated_gt_tre_expertise = -1
                es_pred_expertise = -1
                es_prog_expertise = -1
                es_total_expertise = -1 
                es_tre_expertise = -1

            # Compute incontext variability
            try:
                gt_incontext_variability = self.compute_incontext_variability(T_train, propensities_train)
                es_incontext_variability = self.compute_incontext_variability(T_train, propensities_pred_train) if self.cfg.evaluate_in_context_variability else -1
            except:
                gt_incontext_variability = -1
                es_incontext_variability = -1
            # swap_auroc_1, swap_auprc_1 = self.compute_swap_metrics(T=T_test,
            #                                                     true_outcomes=outcomes_test,
            #                                                     pred_cates=pred_cates,
            #                                                     shap_values_pred=pred_learner_explanations,
            #                                                     shap_base_values_pred=pred_learner_base_values,
            #                                                     k=1)
            # swap_auroc_5, swap_auprc_5 = self.compute_swap_metrics(T=T_test,
            #                                                     true_outcomes=outcomes_test,
            #                                                     pred_cates=pred_cates,
            #                                                     shap_values_pred=pred_learner_explanations,
            #                                                     shap_base_values_pred=pred_learner_base_values,
            #                                                     k=5)
            # swap_auroc_tre, swap_auprc_tre = self.compute_swap_metrics(T=T_test,
            #                                                         true_outcomes=outcomes_test,
            #                                                         pred_cates=pred_cates,
            #                                                         shap_values_pred=pred_learner_explanations,
            #                                                         shap_base_values_pred=pred_learner_base_values,
            #                                                         k=k_tre)

            pred_learner_explanations = np.zeros((self.cfg.simulator.num_T, X_test.shape[0], X_test.shape[1], self.cfg.simulator.dim_Y))
            pred_learner_base_values = np.zeros((self.cfg.simulator.num_T, self.cfg.simulator.dim_Y))
            (
                swap_auroc_all, 
                swap_auprc_all, 
                swap_perc 
            ) = self.compute_swap_metrics(T=T_test,
                                        true_outcomes=outcomes_test,
                                        pred_cates=pred_cates,
                                        shap_values_pred=pred_learner_explanations,
                                        shap_base_values_pred=pred_learner_base_values,
                                        k=k_all)
            
            pred_correct_assignment_precision = self.get_pred_assignment_precision(pred_outcomes, outcomes_test)
            policy_correct_assignment_precision = 1-swap_perc
            # Get scores relative to baseline
            if i == 0:
                baseline_metrics["pehe"] = pehe
                baseline_metrics["pehe_normalized"] = pehe_normalized
                baseline_metrics["factual_outcomes_rmse"] = factual_outcomes_rmse
                baseline_metrics["factual_outcomes_rmse_normalized"] = factual_outcomes_rmse_normalized
                baseline_metrics["cf_outcomes_rmse"] = cf_outcomes_rmse
                baseline_metrics["cf_outcomes_rmse_normalized"] = cf_outcomes_rmse_normalized
                baseline_metrics["swap_auroc_all"] = swap_auroc_all
                baseline_metrics["swap_auprc_all"] = swap_auprc_all

            fc_pehe = pehe_normalized - baseline_metrics["pehe_normalized"]
            fc_factual_outcomes = factual_outcomes_rmse_normalized - baseline_metrics["factual_outcomes_rmse_normalized"] 
            fc_cf_outcomes = cf_outcomes_rmse_normalized - baseline_metrics["cf_outcomes_rmse_normalized"] 
            fc_swap_auroc_all = swap_auroc_all - baseline_metrics["swap_auroc_all"]
            fc_swap_auprc_all = swap_auprc_all - baseline_metrics["swap_auprc_all"]

            # Compute attribution accuracies
            # pred_attribution_est = np.abs(pred_learner_explanations).mean(axis=(0,3)) # average over treatments and outcomes
            # prog_attribution_est = np.abs(prog_learner_explanations).mean(axis=2) # average over outcomes
            # select_attribution_est = np.abs(select_learner_explanations) # average over outcomes

            # pred_acc_scores_all_features = attribution_accuracy(self.all_important_features, pred_attribution_est)
            # pred_acc_scores_predictive_features = attribution_accuracy(self.pred_features, pred_attribution_est)
            # pred_acc_scores_prog_features = attribution_accuracy(self.prog_features, pred_attribution_est)
            # pred_acc_scores_selective_features = attribution_accuracy(self.select_features, pred_attribution_est)

            # prog_acc_scores_all_features = attribution_accuracy(self.all_important_features, prog_attribution_est)
            # prog_acc_scores_predictive_features = attribution_accuracy(self.pred_features, prog_attribution_est)
            # prog_acc_scores_prog_features = attribution_accuracy(self.prog_features, prog_attribution_est)
            # prog_acc_scores_selective_features = attribution_accuracy(self.select_features, prog_attribution_est)

            # select_acc_scores_all_features = attribution_accuracy(self.all_important_features, select_attribution_est)
            # select_acc_scores_predictive_features = attribution_accuracy(self.pred_features, select_attribution_est)
            # select_acc_scores_prog_features = attribution_accuracy(self.prog_features, select_attribution_est)
            # select_acc_scores_selective_features = attribution_accuracy(self.select_features, select_attribution_est)
            
            results_data.append(
                [
                    seed,
                    split_id,
                    x_axis_value,
                    compare_value,
                    model_name,
                    # explainer_names[model_name],
                    true_cates_mean,
                    true_cates_std,
                    pehe,
                    pehe_normalized,
                    factual_outcomes_auroc,
                    cf_outcomes_auroc,
                    rmse_Y0,
                    rmse_Y1,
                    rmse_factual_Y0, 
                    rmse_factual_Y1, 
                    rmse_cf_Y0, 
                    rmse_cf_Y1, 
                    factual_outcomes_rmse,
                    cf_outcomes_rmse,
                    factual_outcomes_rmse_normalized,
                    cf_outcomes_rmse_normalized,
                    factual_outcomes_mean, 
                    cf_outcomes_mean,
                    factual_outcomes_std, 
                    cf_outcomes_std,
                    f_cf_diff,
                    f_cf_diff_norm,
                    fc_pehe,
                    fc_factual_outcomes,
                    fc_cf_outcomes,
                    fc_swap_auroc_all,
                    fc_swap_auprc_all,
                    # swap_auroc_1,
                    # swap_auprc_1,
                    # swap_auroc_5,
                    # swap_auprc_5,
                    # swap_auroc_tre,
                    # swap_auprc_tre,
                    swap_auroc_all,
                    swap_auprc_all,
                    swap_perc,
                    pred_correct_assignment_precision,
                    policy_correct_assignment_precision,
                    ci_coverage,
                    pred_acc_scores_all_features,
                    pred_acc_scores_predictive_features,
                    pred_acc_scores_prog_features,
                    pred_acc_scores_selective_features,
                    prog_acc_scores_all_features,
                    prog_acc_scores_predictive_features,
                    prog_acc_scores_prog_features,
                    prog_acc_scores_selective_features,
                    train_treatment_distribution,
                    test_treatment_distribution,
                    gt_pred_expertise,
                    gt_prog_expertise,
                    gt_tre_expertise,
                    gt_pred_expertise/(gt_pred_expertise + gt_prog_expertise),
                    gt_total_expertise,
                    updated_gt_pred_expertise,
                    updated_gt_prog_expertise,
                    updated_gt_tre_expertise,
                    es_pred_expertise,
                    es_prog_expertise,
                    es_tre_expertise,
                    es_total_expertise,
                    np.abs(gt_pred_expertise - es_pred_expertise),
                    np.abs(gt_prog_expertise - es_prog_expertise),
                    np.abs(gt_total_expertise - es_total_expertise),
                    1-gt_incontext_variability,
                    1-es_incontext_variability,
                    self.training_times[model_name],
                    # select_acc_scores_all_features,
                    # select_acc_scores_predictive_features,
                    # select_acc_scores_prog_features,
                    # select_acc_scores_selective_features
                ]
            )

            metrics_df = pd.DataFrame(
                results_data,
                columns=[
                    "Seed",
                    "Split ID",
                    x_axis_name,
                    compare_name,
                    "Learner",
                    # "Explainer",
                    "CATE true mean",
                    "CATE true std",
                    "PEHE",
                    "Normalized PEHE",
                    "Factual AUROC",
                    "CF AUROC",
                    "RMSE Y0",
                    "RMSE Y1",
                    "Factual RMSE Y0",
                    "Factual RMSE Y1",
                    "CF RMSE Y0",
                    "CF RMSE Y1",
                    "Factual RMSE",
                    "CF RMSE",
                    "Normalized F-RMSE",
                    "Normalized CF-RMSE",
                    "F-Outcome true mean",
                    "CF-Outcome true mean",
                    "F-Outcome true std",
                    "CF-Outcome true std",
                    "F-CF Outcome Diff",
                    "Normalized F-CF Diff",
                    "FC PEHE",
                    "FC F-RMSE",
                    "FC CF-RMSE",
                    "FC Swap AUROC",
                    "FC Swap AUPRC",
                    # "Swap AUROC@1",
                    # "Swap AUPRC@1",
                    # "Swap AUROC@5",
                    # "Swap AUPRC@5",
                    # "Swap AUROC@tre",
                    # "Swap AUPRC@tre",
                    "Swap AUROC@all",
                    "Swap AUPRC@all",
                    "True Swap Perc",
                    "Pred Precision",
                    "Policy Precision",
                    "CI Coverage",
                    "Pred: All features ACC",
                    "Pred: Pred features ACC",
                    "Pred: Prog features ACC",
                    "Pred: Select features ACC",
                    "Prog: All features ACC",
                    "Prog: Pred features ACC",
                    "Prog: Prog features ACC",
                    "Prog: Select features ACC",
                    "T Distribution: Train",
                    "T Distribution: Test",
                    "GT Pred Expertise",
                    "GT Prog Expertise",
                    "GT Tre Expertise",
                    "GT Expertise Ratio",
                    "GT Total Expertise",
                    "Upd. GT Pred Expertise",
                    "Upd. GT Prog Expertise",
                    "Upd. GT Tre Expertise",
                    "ES Pred Expertise",
                    "ES Prog Expertise",
                    "ES Tre Expertise",
                    "ES Total Expertise",
                    "GT-ES Pred Expertise Diff",
                    "GT-ES Prog Expertise Diff",
                    "GT-ES Total Expertise Diff",
                    "GT In-context Var",
                    "ES In-context Var",
                    "Training Duration",
                    # "Select: All features ACC",
                    # "Select: Pred features ACC",
                    # "Select: Prog features ACC",
                    # "Select: Select features ACC",
                ],
            )

            # Save intermediate results
            self.save_results(metrics_df, save_df_only=True, compare_axis_values=None)

        return metrics_df
    
    

       

