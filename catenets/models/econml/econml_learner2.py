from typing import Any, Callable, List

import numpy as np
import torch
from torch import nn

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

# Scores and Models
from econml.dr import DRLearner, SparseLinearDRLearner, LinearDRLearner, ForestDRLearner    
from econml.dml import CausalForestDML, SparseLinearDML, DML, LinearDML
from econml.orf import DMLOrthoForest, DROrthoForest
from econml.iv.dr import SparseLinearDRIV, LinearDRIV, DRIV
from econml.metalearners import XLearner, SLearner, TLearner, DomainAdaptationLearner
from econml.inference import BootstrapInference
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, LogisticRegressionCV, LassoCV
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import LeaveOneOut

# Hydra
from omegaconf import DictConfig

# # Ignore numba warnings
# import warnings
# from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def get_econml_model(cfg: DictConfig, 
                     model_name: str, 
                     discrete_treatment: bool = True, 
                     discrete_outcome: bool = False,
                     seed: int = 123) -> Any:
    
    if cfg.dataset == "cytof_normalized" or cfg.dataset == "cytof_normalized_with_fastdrug" or cfg.debug or cfg.dataset.startswith("melanoma"): #or cfg.dataset == "tcga_100":
        cv = LeaveOneOut()
        # logistic_reg = LogisticRegression(penalty="l1",
        #                                 solver="liblinear",
        #                                 max_iter=5000,
        #                                 random_state=0)
        # reg = Lasso()
    else:
        cv = 5

    logistic_reg = LogisticRegressionCV(penalty='l1',
                                        solver='liblinear',
                                        max_iter=50,
                                        cv=cv,
                                        random_state=seed)
    
    reg = LassoCV(cv=cv, n_alphas=5, max_iter=50)
    #reg = LassoCV(cv=cv, n_alphas=1, max_iter=2)
    #reg = Lasso(alpha=0.1, max_iter=5000, random_state=seed)

    if discrete_outcome:
        reg = logistic_reg

    if cfg.simulator.dim_Y > 1:
        if discrete_outcome:
            #base_model = RandomForestClassifier(n_estimators=100, random_state=seed)
            reg = MultiOutputClassifier(reg)
        else:
            reg = MultiOutputRegressor(reg)
    
    if model_name == "EconML_CausalForestDML":
        est = CausalForestDML(discrete_treatment=discrete_treatment, 
                              discrete_outcome=discrete_outcome, 
                              random_state=seed,
                              **cfg.EconML_CausalForestDML)
        
    elif model_name == "EconML_DML":
        # est = DML(discrete_treatment=discrete_treatment, 
        #           discrete_outcome=discrete_outcome, 
        #           random_state=seed,
        #           **cfg.EconML_DML)
        est = logistic_reg
    
    elif model_name == "EconML_DMLOrthoForest":
        est = DMLOrthoForest(discrete_treatment=discrete_treatment, 
                             random_state=seed,
                             **cfg.EconML_DMLOrthoForest)
    
    elif model_name == "EconML_SparseLinearDML":
        est = SparseLinearDML(discrete_treatment=discrete_treatment, 
                              discrete_outcome=discrete_outcome,
                              random_state=seed,
                              **cfg.EconML_SparseLinearDML)
        
    elif model_name == "EconML_DRLearner":
        est = DRLearner(discrete_outcome=discrete_outcome,
                        random_state=seed,
                        **cfg.EconML_DRLearner)
    
    elif model_name == "EconML_DROrthoForest":
        est = DROrthoForest(random_state=seed,
                            **cfg.EconML_DROrthoForest)

    elif model_name == "EconML_ForestDRLearner":
        est = ForestDRLearner(discrete_outcome=discrete_outcome,
                              random_state=seed,
                              **cfg.EconML_ForestDRLearner)
    
    elif model_name == "EconML_LinearDML":
        est = LinearDML(discrete_treatment=discrete_treatment, 
                        discrete_outcome=discrete_outcome,
                        random_state=seed,
                        **cfg.EconML_LinearDML)
        
    elif model_name == "EconML_LinearDRLearner":
        est = LinearDRLearner(discrete_outcome=discrete_outcome,
                              random_state=seed,
                              **cfg.EconML_LinearDRLearner)
        
    elif model_name == "EconML_SparseLinearDML":
        est = SparseLinearDML(model_y=reg,
                              model_t=logistic_reg,
                              discrete_treatment=discrete_treatment, 
                              discrete_outcome=discrete_outcome,
                              random_state=seed,
                              **cfg.EconML_SparseLinearDML)

    elif model_name == "EconML_SparseLinearDRLearner": # Produces NANs for some reason
        est = SparseLinearDRLearner(model_propensity=logistic_reg,
                                    model_regression=reg,
                                    discrete_outcome=discrete_outcome,
                                    random_state=seed,
                                    **cfg.EconML_SparseLinearDRLearner)
    
    elif model_name == "EconML_XLearner_Lasso":
        est = XLearner(models=reg, 
                       propensity_model=logistic_reg, 
                       cate_models=reg,
                       **cfg.EconML_XLearner_Lasso)

    elif model_name == "EconML_SLearner_Lasso":
        est = SLearner(overall_model=reg,
                       **cfg.EconML_SLearner_Lasso)

    elif model_name == "EconML_TLearner_Lasso":
        est = TLearner(models=reg,
                       **cfg.EconML_TLearner_Lasso)
        
    else:
        raise ValueError(f"Model name {model_name} not recognized")
    
    return est
    
class EconMlEstimator2(BaseCATEEstimator):
    """
    A flexible treatment effect estimator based on the EconML framework.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model_name: str = "DRLearner",
        discrete_treatment: bool = True,
        discrete_outcome: bool = True,
        seed: int = 123,
    ) -> None:
        """
        Initialize the EconML estimator in required format.
        """
        if cfg.evaluate_inference:
            # self.inference_method = "bootstrap"
            if model_name in ["EconML_SparseLinearDRLearner", "EconML_DML", "EconML_DRLearner", "EconML_LinearDML", "EconML_LinearDRLearner", "EconML_ForestDRLearner", "EconML_CausalForestDML"]:
                self.inference_method = "auto"
            elif model_name in ["EconML_SparseLinearDML"]:
                self.inference_method = "debiasedlasso"
            else:
                self.inference_method = BootstrapInference(n_bootstrap_samples=50, n_jobs=-1, verbose=1) #bootstrap_type='percentile',
        else:
            self.inference_method = None

        self.num_T = cfg.simulator.num_T
        self.model_name = model_name
        self.cfg = cfg
        self.average_abs_diff = None

        self.est = get_econml_model(cfg, 
                                    model_name=model_name, 
                                    discrete_treatment=discrete_treatment, 
                                    discrete_outcome=discrete_outcome,
                                    seed=seed)

    def train(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """
        Train the EconML estimator.
        """
        log.info("Training data shapes: X: {}, Y: {}, T: {}".format(X.shape, y.shape, w.shape))
        
        if self.model_name == "EconML_DML":
            self.average_abs_diff = np.abs(np.mean(y[w == 1]) - np.mean(y[w == 0]))
            self.est.fit(X,w)
            return
        # if self.cfg.simulator.dim_Y == 1:
        #     self.est.fit(Y=y.ravel(), T=w, X=X, inference=self.inference_method)
        # else:
        self.est.fit(Y=y, T=w, X=X, inference=self.inference_method)

    # predict function with bool return_po and return potential outcome if true
    def predict(self, X: np.ndarray, T0: np.ndarray = None, T1: np.ndarray = None) -> np.ndarray:
        """
        Predict the treatment effect using the EconML estimator.
        """
        # Deal with case were there are only two treatment options - because of Torch models
        if self.model_name == "EconML_DML":
            prop_pred = self.est.predict(X)
            return torch.Tensor((2*prop_pred-1)*self.average_abs_diff)

        if T0 is None or T1 is None:
            return torch.Tensor(self.est.effect(X))
        
        else:
            return torch.Tensor(self.est.effect(X, T0=T0, T1=T1))
    
    def explain(self, X: np.ndarray, background_samples: np.ndarray = None, explainer_limit: int = None) -> np.ndarray:
        """
        Explain the treatment effect using the EconML estimator.
        """
        if explainer_limit is None:
            explainer_limit = X.shape[0]

        return self.est.shap_values(X[:explainer_limit], background_samples=None)
    
    def infer_effect_ci(self, X: np.ndarray, T0: np.ndarray = None, T1: np.ndarray = None) -> np.ndarray:
        """
        Infer the confidence interval of the treatment effect using the EconML estimator.
        """
        #if self.model_name in ["EconML_SparseLinearDRLearner", "EconML_DML", "EconML_DRLearner", "EconML_LinearDML", "EconML_LinearDRLearner", "EconML_ForestDRLearner", "EconML_CausalForestDML"]:
        if self.num_T > 2:
            log.info("Only the first two treatments are evaluated on the confidence intervals")
        
        cates_conf = self.est.effect_interval(X, alpha=0.05)

        cates_conf_lbs = cates_conf[0]
        cates_conf_ups = cates_conf[1]

        temp = cates_conf_lbs[T0 != 0]
        cates_conf_lbs[T0 != 0] = -cates_conf_ups[T0 != 0]
        cates_conf_ups[T0 != 0] = -temp

        # else:
        #     cates_conf = self.est.effect_interval(X, T0=T0, T1=T1, alpha=0.05)

        #     cates_conf_lbs = cates_conf[0]
        #     cates_conf_ups = cates_conf[1]

        return np.array([cates_conf_lbs, cates_conf_ups])
    
    def predict_outcomes(self, X: np.ndarray, T0: np.ndarray = None, T1: np.ndarray = None, outcomes: np.ndarray = None) -> np.ndarray:
        """
        Predict the potential outcomes using the SLearner and TLearner.
        """
        pred_outcomes = np.zeros((X.shape[0], self.cfg.simulator.num_T, self.cfg.simulator.dim_Y))

        if self.cfg.simulator.num_T > 2 or self.cfg.simulator.dim_Y > 1:
            raise ValueError("Predict outcomes is only supported for dim_Y=1 and num_T=2")
        
        if self.model_name == "EconML_SLearner_Lasso":
            y0_pred = self.est.overall_model.predict(np.hstack([X, np.ones((X.shape[0], 1)),np.zeros((X.shape[0], 1))]))
            y1_pred = self.est.overall_model.predict(np.hstack([X, np.zeros((X.shape[0], 1)),np.ones((X.shape[0], 1))]))
            # pred outcomes shape should be: n, num_T, dim_Y
 
        elif self.model_name == "EconML_TLearner_Lasso":
            y0_pred = self.est.models[0].predict(X)
            y1_pred = self.est.models[1].predict(X)

        elif self.model_name == "EconML_DML":
            props = self.est.predict(X)

            # Fill y0_pred with 0 if prop > 0.5 and 1 otherwise
            y0_pred = np.zeros(X.shape[0])
            y0_pred[props > 0.5] = self.average_abs_diff
            y1_pred = -y0_pred


        pred_outcomes[:, 0, :] = y0_pred.reshape(-1, self.cfg.simulator.dim_Y)
        pred_outcomes[:, 1, :] = y1_pred.reshape(-1, self.cfg.simulator.dim_Y)



        return pred_outcomes
        
    