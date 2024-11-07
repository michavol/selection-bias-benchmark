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
from econml.dr import DRLearner, SparseLinearDRLearner, LinearDRLearner
from econml.dml import CausalForestDML, SparseLinearDML, DML
from econml.orf import DMLOrthoForest
from econml.iv.dr import SparseLinearDRIV, LinearDRIV, DRIV
from econml.metalearners import XLearner, SLearner, TLearner, DomainAdaptationLearner
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression

# # Ignore numba warnings
# import warnings
# from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def get_econml_model(model_name: str, 
                     cv_model_selection, 
                     discrete_treatment=True, 
                     discrete_outcome=True,
                     seed=DEFAULT_SEED, n_jobs=DEFAULT_NJOBS):
    
    if model_name == "EconML_DML":
        est = DML(cv=cv_model_selection, 
                  discrete_treatment=discrete_treatment, 
                  discrete_outcome=discrete_outcome, 
                  random_state=seed)
    
    elif model_name == "EconML_DMLOrthoForest":
        est = DMLOrthoForest(discrete_treatment=discrete_treatment, 
                             n_jobs=n_jobs, random_state=seed)
        
    elif model_name == "EconML_CausalForestDML":
        est = CausalForestDML(cv=cv_model_selection, 
                              discrete_treatment=discrete_treatment, 
                              discrete_outcome=discrete_outcome, 
                              n_jobs=n_jobs, random_state=seed)

    elif model_name == "EconML_SparseLinearDML":
        est = SparseLinearDML(cv=cv_model_selection, 
                              discrete_treatment=discrete_treatment, 
                              discrete_outcome=discrete_outcome, 
                              n_jobs=n_jobs, random_state=seed)
        
    elif model_name == "EconML_SparseLinearDRLearner": # Produces NANs for some reason
        est = SparseLinearDRLearner(cv=cv_model_selection, 
                                    discrete_outcome=discrete_outcome,
                                    n_jobs=n_jobs, random_state=seed)

    elif model_name == "EconML_LinearDRLearner": 
        est = LinearDRLearner(cv=cv_model_selection,
                              discrete_outcome=discrete_outcome,
                              random_state=seed
                              )

    elif model_name == "EconML_DRLearner":
        est = DRLearner(cv=cv_model_selection,
                        discrete_outcome=discrete_outcome,
                        random_state=seed)
    
    elif model_name == "EconML_XLearner":
        est = XLearner(models=Lasso(), propensity_model=LogisticRegression(), cate_models=Lasso())

    elif model_name == "EconML_SLearner":
        est = SLearner(overall_model=Lasso())

    elif model_name == "EconML_TLearner":
        est = TLearner(models=Lasso())
        
    ## IV estimators require instrumental variables, features which only affect the treatment -> all selective features which are not prognostic or predictive
    elif model_name == "EconML_SparseLinearDRIV":
        est = SparseLinearDRIV(cv=cv_model_selection, 
                              discrete_treatment=discrete_treatment, 
                              discrete_outcome=discrete_outcome, 
                              n_jobs=n_jobs, random_state=seed)

    else:
        raise ValueError(f"Model name {model_name} not recognized")
    
    return est
    
class EconMlEstimator(BaseCATEEstimator):
    """
    A flexible treatment effect estimator based on the EconML framework.
    """

    def __init__(
        self,
        model_name: str = "DRLearner",
        cv_model_selection: int = 1,
        discrete_treatment: bool = True,
        discrete_outcome: bool = True,
        n_jobs: int = DEFAULT_NJOBS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.est = get_econml_model(model_name=model_name, 
                                    cv_model_selection=cv_model_selection, 
                                    discrete_treatment=discrete_treatment, 
                                    discrete_outcome=discrete_outcome, 
                                    seed=seed, n_jobs=n_jobs)

    def train(self, X: np.ndarray, y: np.ndarray, T: np.ndarray) -> None:
        self.est.fit(Y=y, T=T, X=X)

    # predict function with bool return_po and return potential outcome if true
    def predict(self, X: np.ndarray) -> np.ndarray:
        return torch.Tensor(self.est.effect(X)).ravel()
    
    