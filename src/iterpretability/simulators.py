# stdlib
import random
from typing import Tuple
import src.iterpretability.logger as log

# third party
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import zscore
from omegaconf import DictConfig, OmegaConf
from src.iterpretability.utils import enable_reproducible_results
from abc import ABC, abstractmethod
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For computing the propensities from scores
from scipy.special import softmax
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

EPS = 0
class SimulatorBase(ABC):
    """
    Base class for simulators.
    """
    @abstractmethod
    def simulate(self, X: np.ndarray, outcomes: np.ndarray = None) -> Tuple:
        raise NotImplementedError
    
    @abstractmethod
    def get_simulated_data(self, train_ratio: float) -> Tuple:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def selective_features(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def prognostic_features(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def predictive_features(self) -> np.ndarray:
        raise NotImplementedError
    
class TYSimulator(SimulatorBase):
    """
    Data generation process class for simulating treatment selection and outcomes (and effects)
    """
    nonlinear_fcts = [
                #lambda x: np.abs(x),
                lambda x: np.exp(-(x**2) / 2),
                #  lambda x: 1 / (1 + x**2),
                # lambda x: np.sqrt(x)*(1+x),
                #lambda x: np.cos(5*x),
                #lambda x: x**2,
                # lambda x: np.arctan(x),
                # lambda x: np.tanh(x),
                # lambda x: np.sin(x),
                # lambda x: np.log(1 + x**2),
                #lambda x: np.sqrt(0.02 + x**2),
                #lambda x: np.cosh(x),
            ]
    
    def __init__(
        self,
        # Data dimensionality
        dim_X: int,

        # Seed
        seed: int = 42,

        # Simulation type
        simulation_type: str = "ty",

        # Dimensionality of treatments and outcome
        num_binary_outcome: int = 0,
        outcome_unbalancedness_ratio: float = 0,
        standardize_outcome: bool = False,
        num_T: int = 3,
        dim_Y: int = 3,

        # Scale parameters
        predictive_scale: float = 1,
        prognostic_scale: float = 1,
        propensity_scale: float = 1,
        unbalancedness_exp: float = 0,
        nonlinearity_scale: float = 1,
        propensity_type: str = "prog_pred",
        alpha: float = 0.5,
        enforce_balancedness: bool = False,

        # Control
        include_control: bool = False,

        # Important features
        num_pred_features: int = 5,
        num_prog_features: int = 5,
        num_select_features: int = 5,
        feature_type_overlap: str = "sel_none",
        treatment_feature_overlap: bool = False,

        # Feature selection
        random_feature_selection: bool = False,
        nonlinearity_selection_type: bool = True,

        # Noise
        noise: bool = True,
        noise_std: float = 0.1,
        
    ) -> None:
        # Number of features
        self.dim_X = dim_X

        # Make sure results are reproducible by setting seed for np, torch, random
        self.seed = seed
        enable_reproducible_results(seed=self.seed)

        # Simulation type
        self.simulation_type = simulation_type

        # Store dimensions
        self.num_binary_outcome = num_binary_outcome
        self.outcome_unbalancedness_ratio = outcome_unbalancedness_ratio
        self.standardize_outcome = standardize_outcome
        self.num_T = num_T
        self.dim_Y = dim_Y

        # Scale parameters
        self.predictive_scale = predictive_scale
        self.prognostic_scale = prognostic_scale
        self.propensity_scale = propensity_scale
        self.unbalancedness_exp = unbalancedness_exp
        self.nonlinearity_scale = nonlinearity_scale
        self.propensity_type = propensity_type
        self.alpha = alpha
        self.enforce_balancedness = enforce_balancedness

        # Control
        self.include_control = include_control

        # Important features
        self.num_pred_features = num_pred_features
        self.num_prog_features = num_prog_features
        self.num_select_features = num_select_features
        self.num_important_features = self.num_T*(num_pred_features + num_select_features) + num_prog_features
        self.feature_type_overlap = feature_type_overlap
        self.treatment_feature_overlap = treatment_feature_overlap

        # Feature selection
        self.random_feature_selection = random_feature_selection
        self.nonlinearity_selection_type = nonlinearity_selection_type

        # Noise
        self.noise = noise
        self.noise_std = noise_std

        # Setup variables
        self.nonlinearities = None
        self.prog_mask, self.pred_masks, self.select_masks = None, None, None
        self.prog_weights, self.pred_weights, self.select_weights = None, None, None

        # Setup
        self.setup()

        # Simulation variables
        self.X = None
        self.prog_scores, self.pred_scores, self.select_scores = None, None, None
        self.select_scores_pred_overlap = None
        self.select_scores_prog_overlap = None
        self.propensities, self.outcomes, self.T, self.Y = None, None, None, None
    
    def get_simulated_data(self):
        """
        Extract results and split into training and test set. Include counterfactual outcomes and propensities.
        """
        return self.X, self.T, self.Y, self.outcomes, self.propensities

        ## OLD CODE
        # Split data
        # train_size = int(train_ratio * self.X.shape[0])

        # if self.num_binary_outcome > 0:
        #     (
        #         X_train, X_test, 
        #         Y_train, Y_test, 
        #         T_train, T_test,
        #         outcomes_train, outcomes_test,
        #         propensities_train, propensities_test,
        #     ) = train_test_split(self.X, self.Y, self.T, self.outcomes, self.propensities, train_size=train_size, stratify=self.Y)
        # else:
        #     X_train, X_test = self.X[:train_size], self.X[train_size:]
        #     T_train, T_test = self.T[:train_size], self.T[train_size:]
        #     Y_train, Y_test = self.Y[:train_size], self.Y[train_size:]

        #     outcomes_train, outcomes_test = self.outcomes[:train_size,:,:], self.outcomes[train_size:,:,:]
        #     propensities_train, propensities_test = self.propensities[:train_size], self.propensities[train_size:]

        # if train_ratio == 1:
        #     return self.X, self.T, self.Y, self.outcomes, self.propensities
        
        # return X_train, X_test, T_train, T_test, Y_train, Y_test, outcomes_train, outcomes_test, propensities_train, propensities_test

    def simulate(self, X, outcomes=None) -> Tuple:
        """
        Simulate treatment and outcome for a dataset based on the configuration.
        """
        log.debug(
            f'Simulating treatment and outcome for a dataset with:'
            f'\n==================================================================='
            f'\nDim X: {self.dim_X}'
            f'\nDim T: {self.num_T}'
            f'\nDim Y: {self.dim_Y}'
            f'\nPredictive Scale: {self.predictive_scale}'
            f'\nPrognostic Scale: {self.prognostic_scale}'
            f'\nPropensity Scale: {self.propensity_scale}'
            f'\nUnbalancedness Exponent: {self.unbalancedness_exp}'
            f'\nNonlinearity Scale: {self.nonlinearity_scale}'
            f'\nNum Pred Features: {self.num_pred_features}'
            f'\nNum Prog Features: {self.num_prog_features}'
            f'\nNum Select Features: {self.num_select_features}'
            f'\nFeature Overlap: {self.treatment_feature_overlap}'
            f'\nRandom Feature Selection: {self.random_feature_selection}'
            f'\nNonlinearity Selection Type: {self.nonlinearity_selection_type}'
            f'\nNoise: {self.noise}'
            f'\nNoise Std: {self.noise_std}'
            f'\n===================================================================\n'
        )

        # 1. Store data with min max scaling to range [0, 1]
        self.X = X
        # self.X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + EPS)

        # 2. Compute scores for prognostic, predictive, and selective features
        self.compute_scores()

        # 3. Compute factual and counterfactual outcomes based on the data and the predictive and prognostic scores
        self.compute_all_outcomes()

        # 4. Compute propensities based on the data and the selective scores
        self.compute_propensities()

        # 5. Sample treatment assignment based on the propensities
        self.sample_T()

        # 6. Extract the outcome based on the treatment assignment
        self.extract_Y()

        return None
    
    def setup(self) -> None:
        """
        Setup the simulator by defining variables which remain the same across simulations with different samples but the same configuration.
        """
        # 1. Sample nonlinearities used 
        num_nonlinearities = 2 + self.dim_Y # Different non-linearities for each outcome (predictive), same for all treatments
        self.nonlinearities = self.sample_nonlinearities(num_nonlinearities)

        # 2. Set important feature masks - determine which features should be used for treatment selection, outcome prediction
        self.sample_important_feature_masks()

        # 3. Sample weights for features
        self.sample_uniform_weights()

    def get_true_cates(self, 
                       X: np.ndarray, 
                       T: np.ndarray, 
                       outcomes: np.ndarray) -> np.ndarray:
        """
        Compute true CATEs for each treatment based on the data and the outcomes.
        Always use the selected treatment as the base treatment.
        """
        # Compute CATEs for each treatment
        cates = np.zeros((X.shape[0], self.num_T, self.dim_Y))

        for i in range(X.shape[0]):
            for j in range(self.num_T):
                cates[i,j,:] = outcomes[i,j,:] - outcomes[i,int(T[i]),:]

        log.debug(
            f'\nCheck if true CATEs are computed correctly:'
            f'\n==================================================================='
            f'\nOutcomes: {outcomes.shape}'
            f'\n{outcomes}'
            f'\n\nTreatment Assignment: {T.shape}'
            f'\n{T}'
            f'\n\nTrue CATEs: {cates.shape}'
            f'\n{cates}'
            f'\n===================================================================\n'
        )

        return cates
    
    def extract_Y(self) -> None:
        """
        Extract the outcome based on the treatment assignment.
        """
        self.Y = self.outcomes[np.arange(self.X.shape[0]), self.T] 

        log.debug(
            f'\nCheck if outcomes are extracted correctly:'
            f'\n==================================================================='
            f'\nOutcomes'
            f'\n{self.outcomes}'
            f'\n{self.outcomes.shape}'
            f'\n\nTreatment Assignment'
            f'\n{self.T}'
            f'\n{self.T.shape}'
            f'\n\nExtracted Outcomes'
            f'\n{self.Y}'
            f'\n{self.Y.shape}'
            f'\n===================================================================\n'
        )

        return None

    def compute_all_outcomes_toy(self) -> None:
        # Compute outcomes for each treatment and outcome
        outcomes = np.zeros((self.X.shape[0], self.num_T, self.dim_Y))
        X0 = self.X[:,0]
        X1 = self.X[:,1]

        k=20
        nonlinearity = lambda x: 1 / (1 + np.exp(-k * (x - 0.5))) #logistic
        
        if self.propensity_type.startswith("toy1") or self.propensity_type.startswith("toy3") or self.propensity_type.startswith("toy4"):
            fun_y0 = lambda X0, X1: X0
            fun_y1 = lambda X0, X1: 1-X0

        elif self.propensity_type.startswith("toy2"):
            fun_y0 = lambda X0, X1: X0
            fun_y1 = lambda X0, X1: 1-X1

        elif self.propensity_type.startswith("toy6"):
            fun_y0 = lambda X0, X1: X0
            fun_y1 = lambda X0, X1: X1

        elif self.propensity_type.startswith("toy5"):
            fun_y0 = lambda X0, X1: np.sin(X0*10*np.pi)
            fun_y1 = lambda X0, X1: np.sin((1-X0)*10*np.pi)

        elif self.propensity_type.startswith("toy7"):
            fun_y0 = lambda X0, X1: nonlinearity(X0)-nonlinearity(X1)
            fun_y1 = lambda X0, X1: nonlinearity(X0)+nonlinearity(X1)

        elif self.propensity_type.startswith("toy8"):
            fun_y0 = lambda X0, X1: X0
            fun_y1 = lambda X0, X1: 1-X0

        Y = np.array([fun_y0(X0, X1),fun_y1(X0, X1)]).T

        if self.propensity_type.endswith("nonlinear"):
            Y = nonlinearity(Y)

        Y = zscore(Y, axis=None)

        outcomes[:,:,0] = Y


        return outcomes

    def compute_all_outcomes(self) -> None:
        """
        Compute factual and counterfactual outcomes based on the data and the predictive and prognostic scores.
        """
        if self.propensity_type.startswith("toy"):
            outcomes = self.compute_all_outcomes_toy()

        else:
            # Compute outcomes for each treatment and outcome
            outcomes = np.zeros((self.X.shape[0], self.num_T, self.dim_Y))

            for i in range(self.num_T):
                for j in range(self.dim_Y):
                    if self.include_control and i == 0:
                        outcomes[:,i,j] = self.prognostic_scale*self.prog_scores[:,j]

                    else:
                        outcomes[:,i,j] = self.prognostic_scale*self.prog_scores[:,j] + self.predictive_scale*self.pred_scores[:,i,j]

        # Add gaussian noise to outcomes
        if self.noise:
            outcomes = outcomes + np.random.normal(0, self.noise_std, size=outcomes.shape)

        # Create binary outcomes and introduce unbalancedness
        if int(self.num_binary_outcome) > 0:
            for j in range(self.num_binary_outcome):
                scores = zscore(outcomes[:,:,j], axis=0)
                prob = expit(scores)
                outcomes[:,:,j] = prob > self.outcome_unbalancedness_ratio
                
        self.outcomes = outcomes

        # Standardize outcomes
        if self.standardize_outcome:
            # normalize outcomes per outcome
            self.outcomes = zscore(self.outcomes, axis=0)

        log.debug(
            f'\nCheck if outcomes are computed correctly:'
            f'\n==================================================================='
            f'\nProg Scores'
            f'\n{self.prog_scores}'
            f'\n{self.prog_scores.shape}'
            f'\n\nPred Scores'
            f'\n{self.pred_scores}'
            f'\n{self.pred_scores.shape}'
            f'\n\nOutcomes'
            f'\n{self.outcomes}'
            f'\n{self.outcomes.shape}'
            f'\n\nMean Outcomes'
            f'\n{self.outcomes.mean(axis=0)}'
            f'\n\nVariance Outcomes'
            f'\n{self.outcomes.var(axis=0)}'
            f'\n===================================================================\n'
        )

        return None


    def sample_T(self) -> None:
        """
        Sample treatment assignment based on the propensities.
        """
        # Sample from the resulting categorical distribution per row
        self.T = np.array([np.random.choice([tre for tre in range(self.propensities.shape[1])], p=row) for row in self.propensities])

        log.debug(
            f'\nCheck if treatment assignment is sampled correctly:'
            f'\n==================================================================='
            f'\nPropensities'
            f'\n{self.propensities}'
            f'\n{self.propensities.shape}'
            f'\n\nTreatment Assignment'
            f'\n{self.T}'
            f'\n{self.T.shape}'
            f'\n\nUnique Treatment Counts'
            f'\n{np.unique(self.T, return_counts=True)}'
            f'\n===================================================================\n'
        )

        return None
    
    def get_unbalancedness_weights(self, size: int) -> np.ndarray:
        """
        Create weights for introducing unbalancedness for class probabilities.
        """
        # Sample initial distribution of treatment assignment 
        unb_weights = np.random.uniform(0, 1, size=size) 
        unb_weights = unb_weights / unb_weights.sum()

        # Standardize the weights and make sure that a treatment doesn't completely disappear for small unbalancedness exponents
        min_val = unb_weights.min()
        range_val = unb_weights.max() - min_val
        unb_weights = (unb_weights - min_val) / range_val
        unb_weights = 0.01 + unb_weights * 0.98

        return unb_weights
    
    def compute_propensity_scores_toy(self) -> np.ndarray:
        X0 = self.X[:,0]
        X1 = self.X[:,1]

        if self.propensity_type.startswith("toy1"):
            fun_t0 = lambda X0, X1: X0
            fun_t1 = lambda X0, X1: 1-X0

        elif self.propensity_type.startswith("toy2"):
            fun_t0 = lambda X0, X1: X0
            fun_t1 = lambda X0, X1: 1-X1

        elif self.propensity_type.startswith("toy3"):
            fun_t0 = lambda X0, X1: X1
            fun_t1 = lambda X0, X1: 1-X1

        elif self.propensity_type.startswith("toy4"):
            fun_t0 = lambda X0, X1: np.sin(X0*10*np.pi)
            fun_t1 = lambda X0, X1: np.sin((1-X0)*10*np.pi)

        elif self.propensity_type.startswith("toy5"):
            fun_t0 = lambda X0, X1: 1-X0
            fun_t1 = lambda X0, X1: X0

        elif self.propensity_type.startswith("toy6"):
            fun_t0 = lambda X0, X1: 1-X0
            fun_t1 = lambda X0, X1: X0

        elif self.propensity_type.startswith("toy7"):
            fun_t0 = lambda X0, X1: 1-X0
            fun_t1 = lambda X0, X1: X0

        elif self.propensity_type.startswith("toy8"):
            fun_t0 = lambda X0, X1: 1-X0
            fun_t1 = lambda X0, X1: X0

        scores = np.array([fun_t0(X0, X1),fun_t1(X0, X1)]).T

        return scores


    def compute_propensities(self) -> None:
        """
        Compute propensities based on the data and the selective scores.
        """
        
        select_scores_pred_overlap = zscore(self.select_scores_pred_overlap, axis=0) # Comment for Predictive Epertise
        select_scores_prog_overlap = zscore(self.select_scores_prog_overlap, axis=0) # Comment for Predictive Epertise
        select_scores_none = zscore(self.select_scores, axis=0) # Comment for Predictive Epertise

        select_scores_pred = np.zeros((self.X.shape[0], self.num_T))
        select_scores_pred_flipped = np.zeros((self.X.shape[0], self.num_T))
        select_scores_prog = np.zeros((self.X.shape[0], self.num_T))
        select_scores_tre = np.zeros((self.X.shape[0], self.num_T))

        select_scores_pred[:,0] = self.outcomes[:,0,0] - self.outcomes[:,1,0]
        select_scores_pred[:,1] = self.outcomes[:,1,0] - self.outcomes[:,0,0]

        select_scores_pred_flipped[:,0] = self.outcomes[:,1,0] - self.outcomes[:,0,0]
        select_scores_pred_flipped[:,1] = self.outcomes[:,0,0] - self.outcomes[:,1,0]

        select_scores_prog[:,0] = self.outcomes[:,0,0]
        select_scores_prog[:,1] = -self.outcomes[:,0,0]

        select_scores_tre[:,0] = -self.outcomes[:,1,0]
        select_scores_tre[:,1] = self.outcomes[:,1,0]

        if self.propensity_type == "prog_tre":
            scores = self.alpha * select_scores_tre + (1 - self.alpha) * select_scores_prog

        # Standardize all scores
        select_scores_pred = zscore(select_scores_pred, axis=0)
        select_scores_pred_flipped = zscore(select_scores_pred_flipped, axis=0)
        select_scores_prog = zscore(select_scores_prog, axis=0)
        select_scores_tre = zscore(select_scores_tre, axis=0)

        if self.propensity_type == "prog_pred":
            scores = self.alpha * select_scores_pred + (1 - self.alpha) * select_scores_prog

        elif self.propensity_type == "prog_tre":
            pass

        elif self.propensity_type == "none_prog":
            scores = self.alpha * select_scores_prog + (1 - self.alpha) * select_scores_none

        elif self.propensity_type == "none_pred":
            scores = self.alpha * select_scores_pred + (1 - self.alpha) * select_scores_none

        elif self.propensity_type == "none_tre":
            scores = self.alpha * select_scores_tre + (1 - self.alpha) * select_scores_none

        elif self.propensity_type == "none_pred_flipped":
            scores = self.alpha * select_scores_pred_flipped + (1 - self.alpha) * select_scores_none

        elif self.propensity_type == "pred_pred_flipped":
            scores = self.alpha * select_scores_pred_flipped + (1 - self.alpha) * select_scores_pred

        elif self.propensity_type == "none_pred_overlap":
            scores = self.alpha * select_scores_pred_overlap + (1 - self.alpha) * select_scores_none
            
        elif self.propensity_type == "none_prog_overlap":
            scores = self.alpha * select_scores_prog_overlap + (1 - self.alpha) * select_scores_none

        elif self.propensity_type == "pred_overalp_prog_overlap":
            scores = self.alpha * select_scores_prog_overlap + (1 - self.alpha) * select_scores_pred_overlap

        elif self.propensity_type == "rct_none":
            scores = select_scores_none

        elif self.propensity_type.startswith("toy"):
            scores = self.compute_propensity_scores_toy()

        else:
            raise ValueError(f"Unknown propensity type {self.propensity_type}.")

        if self.enforce_balancedness:
            scores = zscore(scores, axis=0)

        if self.propensity_type == "rct_none":
            scores = self.alpha * select_scores_none

        # Introduce unbalancedness and manipulate unbalancedness weights for comparable experiments with different seeds
        unb_weights = self.get_unbalancedness_weights(size=scores.shape[1])

        # Apply the softmax function to each row to get probabilities
        p = softmax(self.propensity_scale*scores, axis=1)

        # Scale probabilities to introduce unbalancedness
        p = p * (1 - unb_weights) ** self.unbalancedness_exp

        # Make sure rows add up to one again
        row_sums = p.sum(axis=1, keepdims=True)
        p = p / row_sums
        self.propensities = p

        log.debug(
            f'\nCheck if propensities are computed correctly:'
            f'\n==================================================================='
            f'\nSelect Scores'
            f'\n{self.select_scores}'
            f'\n{self.select_scores.shape}'
            f'\n\nPropensities'
            f'\n{self.propensities}'
            f'\n{self.propensities.shape}'
            f'\n===================================================================\n'
        )

        return None

    def compute_scores(self) -> None:
        """
        Compute scores for prognostic, predictive, and selective features based on the data and the feature weights.
        """
        # Each column of the score matrix corresponds to the score for a specific outcome. Rows correspond to samples.
        prog_lin = self.X @ self.prog_weights.T
        select_lin = self.X @ self.select_weights.T
        select_lin_pred = self.X @ self.select_weights_pred.T
        select_lin_prog = self.X @ self.select_weights_prog.T

        log.debug(
            f'\nCheck if linear scores are computed correctly for selective features:'
            f'\n==================================================================='
            f'\nself.X'
            f'\n{self.X}'
            f'\n{self.X.shape}'
            f'\n\nSelect Weights'
            f'\n{self.select_weights}'
            f'\n{self.select_weights.shape}'
            f'\n\nSelect Lin'
            f'\n{select_lin}'
            f'\n{select_lin.shape}'
            f'\n===================================================================\n'
        )

        # Compute scores for predictive and selective features for each treatment and outcome
        pred_lin = np.zeros((self.X.shape[0], self.num_T, self.dim_Y))

        # This creates a score for each treatment and outcome for each sample
        for i in range(self.num_T):
            pred_lin[:,i,:] = self.X @ self.pred_weights[i].T
        
        # Introduce non-linearity and get final scores
        prog_scores = (1 - self.nonlinearity_scale) * prog_lin + self.nonlinearity_scale * self.nonlinearities[0](prog_lin)
        select_scores = (1 - self.nonlinearity_scale) * select_lin + self.nonlinearity_scale * self.nonlinearities[1](select_lin)
        select_scores_pred_overlap = (1 - self.nonlinearity_scale) * select_lin_pred + self.nonlinearity_scale * self.nonlinearities[1](select_lin_pred)
        select_scores_prog_overlap = (1 - self.nonlinearity_scale) * select_lin_prog + self.nonlinearity_scale * self.nonlinearities[1](select_lin_prog)

        pred_scores = np.zeros((self.X.shape[0], self.num_T, self.dim_Y))
        for i in range(self.dim_Y):
            pred_scores[:,:,i] = (1 - self.nonlinearity_scale) * pred_lin[:,:,i] + self.nonlinearity_scale * self.nonlinearities[i+2](pred_lin[:,:,i])

        log.debug(
            f'\nCheck if all scores are computed correctly for predictive features:'
            f'\n==================================================================='
            f'\nself.X'
            f'\n{self.X}'
            f'\n{self.X.shape}'
            f'\n\nPred Weights'
            f'\n{self.pred_weights}'
            f'\n{self.pred_weights.shape}'
            f'\n\nPred Lin'
            f'\n{pred_lin}'
            f'\n{pred_lin.shape}'
            f'\n\nPred Scores'
            f'\n{pred_scores}'
            f'\n{pred_scores.shape}'
            f'\n===================================================================\n'
        )

        self.prog_scores = prog_scores
        self.select_scores = select_scores
        self.select_scores_pred_overlap = select_scores_pred_overlap
        self.select_scores_prog_overlap = select_scores_prog_overlap

        self.pred_scores = pred_scores

        return None
    
    @property
    def weights(self) -> Tuple:
        """
        Return weights for prognostic, predictive, and selective features.
        """
        return self.prog_weights, self.pred_weights, self.select_weights
    
    def sample_uniform_weights(self) -> None:
        """
        sample uniform weights for the features.
        """
        if self.propensity_type.startswith("toy"):
            self.prog_weights = np.zeros((self.dim_Y, self.dim_X))
            self.pred_weights = np.zeros((self.num_T, self.dim_Y, self.dim_X))
            self.select_weights = np.zeros((self.num_T, self.dim_X))
            self.select_weights_pred = np.zeros((self.num_T, self.dim_X))
            self.select_weights_prog = np.zeros((self.num_T, self.dim_X))
            return None


        # Sample weights for prognostic features, a weight for every outcome
        prog_weights = np.random.uniform(-1, 1, size=(self.dim_Y, self.dim_X)) * self.prog_mask

        # Sample weights for predictive and selective features, a weight for every dimension for every treatment and outcome
        pred_weights = np.random.uniform(-1, 1, size=(self.num_T, self.dim_Y, self.dim_X))
        select_weights = np.random.uniform(-1, 1, size=(self.num_T, self.dim_X))
        select_weights_pred = select_weights.copy()
        select_weights_prog = select_weights.copy()

        # # Sample weights for prognostic features, a weight for every outcome
        # prog_weights = np.random.uniform(0, 1, size=(self.dim_Y, self.dim_X)) * self.prog_mask

        # # Sample weights for predictive and selective features, a weight for every dimension for every treatment and outcome
        # pred_weights = np.random.uniform(0, 1, size=(self.num_T, self.dim_Y, self.dim_X))
        # select_weights = np.random.uniform(0, 1, size=(self.num_T, self.dim_X))

        # # Make sure treatments are different
        # pred_weights[0] = -pred_weights[0]
        # select_weights[0] = -select_weights[0]

        # # Ones as weights
        # prog_weights = np.ones((self.dim_Y, self.dim_X)) * self.prog_mask#/ self.prog_mask.sum()
        # pred_weights = np.ones((self.num_T, self.dim_Y, self.dim_X))  #/ self.pred_masks.sum(axis=1, keepdims=True)
        # select_weights = np.ones((self.num_T, self.dim_X))  #/ self.select_masks.sum(axis=1, keepdims=True)

        # Mask weights for features that are not important
        for i in range(self.num_T):
            pred_weights[i] = pred_weights[i] * self.pred_masks[:,i]
            select_weights[i] = select_weights[i] * self.select_masks[:,i]
            select_weights_pred[i] = select_weights_pred[i] * self.select_masks_pred[:,i]
            select_weights_prog[i] = select_weights_prog[i] * self.select_masks_prog[:,i]

        # for i in range(self.num_T):
        #     row_sums = pred_weights[i].sum(axis=1, keepdims=True)
        #     pred_weights[i] = pred_weights[i] / row_sums

        #     row_sums = select_weights[i].sum()
        #     select_weights[i] = select_weights[i] / row_sums

        # # Make sure that prog weights sum to one per outcome
        # row_sums = prog_weights.sum(axis=1, keepdims=True)
        # prog_weights = prog_weights / row_sums

        log.debug(
            f'\nCheck if masks are applied correctly:'
            f'\n==================================================================='
            f'\nSelect Weights'
            f'\n{select_weights}'
            f'\n{select_weights.shape}'
            f'\n\nSelect Masks'
            f'\n{self.select_masks}'
            f'\n{self.select_masks.shape}'
            f'\n\nPred Weights'
            f'\n{pred_weights}'
            f'\n{pred_weights.shape}'
            f'\n\nPred Masks'
            f'\n{self.pred_masks}'
            f'\n{self.pred_masks.shape}'
            f'\n===================================================================\n'
        )
        
        self.prog_weights = prog_weights
        self.pred_weights = pred_weights
        self.select_weights = select_weights
        self.select_weights_pred = select_weights_pred
        self.select_weights_prog = select_weights_prog

        return None
    
    @property
    def all_important_features(self) -> np.ndarray:
        """
        Return all important feature indices.
        """
        all_important_features = np.union1d(self.predictive_features, self.prognostic_features)
        all_important_features = np.union1d(all_important_features, self.selective_features)

        log.debug(
            f'\nCheck if all important features are computed correctly:'
            f'\n==================================================================='
            f'\nProg Features'
            f'\n{self.prognostic_features}'
            f'\n\nPred Features'
            f'\n{self.predictive_features}'
            f'\n\nSelect Features'
            f'\n{self.selective_features}'
            f'\n\nAll Important Features'
            f'\n{all_important_features}'
            f'\n===================================================================\n'
        )

        return all_important_features

    @property
    def prognostic_features(self) -> np.ndarray:
        """
        Return prognostic feature indices.
        """
        prog_features = np.where((self.prog_mask).astype(np.int32) != 0)
        return prog_features
    
    @property
    def predictive_features(self) -> np.ndarray:
        """
        Return predictive feature indices.
        """
        pred_features = np.where((self.pred_masks.sum(axis=1)).astype(np.int32) != 0)
        return pred_features

    @property
    def selective_features(self) -> np.ndarray:
        """
        Return selective feature indices.
        """
        select_features = np.where((self.select_masks.sum(axis=1)).astype(np.int32) != 0)
        return select_features
    
    def sample_important_feature_masks(self) -> None:
        """
        Pick features that are important for treatment selection, outcome prediction, and prognostic prediction based on the configuration.
        """
        if self.propensity_type.startswith("toy"):
            self.prog_mask = np.zeros(shape=(self.dim_X))
            self.pred_masks = np.zeros(shape=(self.dim_X, self.num_T))
            self.select_masks = np.zeros(shape=(self.dim_X, self.num_T))

            self.prog_mask[0] = 1
            self.pred_masks[0,0] = 1
            self.pred_masks[1,1] = 1
            self.select_masks[0,0] = 1
            self.select_masks[1,1] = 1

            return None

        # Get indices for features and shuffle if random_feature_selection is True
        all_indices = np.arange(self.dim_X)
        n = self.num_pred_features

        if self.random_feature_selection:
            np.random.shuffle(all_indices)

        # Initialize masks
        prog_mask = np.zeros(shape=(self.dim_X))
        pred_masks = np.zeros(shape=(self.dim_X, self.num_T))
        select_masks = np.zeros(shape=(self.dim_X, self.num_T))

        # Handle case with feature overlap
        if self.feature_type_overlap == "sel_pred":
        
            prog_indices = all_indices[:n]
            prog_mask[prog_indices] = 1

            if self.treatment_feature_overlap:
                assert 2*n <= int(self.dim_X)
                pred_indices = np.array(self.num_T * [all_indices[n:2*n]])
                select_indices = np.array(self.num_T * [all_indices[n:2*n]])

                prog_mask[prog_indices] = 1
                pred_masks[pred_indices] = 1
                select_masks[select_indices] = 1

            else:
                assert n*(1+self.num_T) <= int(self.dim_X)
                for i in range(self.num_T):
                    pred_indices = all_indices[(i+1)*n: (i+2)*n]
                    select_indices = all_indices[(i+1)*n: (i+2)*n]
            
                    pred_masks[pred_indices,i] = 1
                    select_masks[select_indices,i] = 1

        elif self.feature_type_overlap == "sel_prog":

            if self.treatment_feature_overlap:
                assert 2*n <= int(self.dim_X)
                prog_indices = all_indices[:n]
                prog_mask[prog_indices] = 1
                pred_indices = np.array(self.num_T * [all_indices[n:2*n]])
                select_indices = np.array(self.num_T * [all_indices[:n]])

                prog_mask[prog_indices] = 1
                pred_masks[pred_indices] = 1
                select_masks[select_indices] = 1

            else:
                assert 2*n*self.num_T <= int(self.dim_X)
                prog_indices = all_indices[:n*self.num_T:self.num_T]
                prog_mask[prog_indices] = 1
                for i in range(self.num_T):
                    select_indices = all_indices[i*n: (i+1)*n]
                    pred_indices = all_indices[(i+self.num_T+1)*n: (i+self.num_T+2)*n]
            
                    pred_masks[pred_indices,i] = 1
                    select_masks[select_indices,i] = 1

        elif self.feature_type_overlap == "sel_none":
            prog_indices = all_indices[:n]
            prog_mask[prog_indices] = 1

            if self.treatment_feature_overlap:
                assert 3*n <= int(self.dim_X)
                pred_indices = np.array(self.num_T * [all_indices[n:2*n]])
                select_indices = np.array(self.num_T * [all_indices[2*n:3*n]])

                prog_mask[prog_indices] = 1
                pred_masks[pred_indices] = 1
                select_masks[select_indices] = 1

            else:
                #assert n+2*n*self.num_T <= int(self.dim_X)
                for i in range(1,self.num_T+1):
                    select_indices = all_indices[i*n: (i+1)*n]
                    pred_indices = all_indices[(i+self.num_T)*n: (i+self.num_T+1)*n]
                    pred_masks[pred_indices,i-1] = 1
                    select_masks[select_indices,i-1] = 1

        # # Handle case with feature overlap
        # if self.feature_overlap:
        #     assert max(self.num_pred_features, self.num_prog_features, self.num_select_features) <= int(self.dim_X)
        
        #     prog_indices = all_indices[:self.num_prog_features]
        #     pred_indices = np.array(self.num_T * [all_indices[:self.num_pred_features]])
        #     select_indices = np.array(self.num_T * [all_indices[:self.num_select_features]])

        #     prog_mask[prog_indices] = 1
        #     pred_masks[pred_indices] = 1
        #     select_masks[select_indices] = 1

        # # Handle case without feature overlap
        # else:
        #     assert (self.num_prog_features + self.num_T * (self.num_pred_features + self.num_select_features)) <= int(self.dim_X)

        #     prog_indices = all_indices[:self.num_prog_features]
        #     prog_mask[prog_indices] = 1
        #     pred_indices = all_indices[self.num_prog_features : (self.num_prog_features + self.num_T*self.num_pred_features)]
        #     select_indices = all_indices[(self.num_prog_features + self.num_T*self.num_pred_features):(self.num_prog_features + self.num_T*(self.num_pred_features+self.num_select_features))]
            
        #     # Mask features for every treatment
        #     for i in range(self.num_T):
        #         pred_masks[pred_indices[i*self.num_pred_features:(i+1)*self.num_pred_features],i] = 1
        #         select_masks[select_indices[i*self.num_select_features:(i+1)*self.num_select_features],i] = 1

        self.prog_mask = prog_mask
        self.pred_masks = pred_masks
        self.select_masks = select_masks
        self.select_masks_pred = pred_masks.copy()
        self.select_masks_prog = select_masks.copy()

        log.debug(
            f'\nCheck if important features are sampled correctly:'
            f'\n==================================================================='
            f'\nProg Indices'
            f'\n{prog_indices}'
            f'\n\nPred Indices'
            f'\n{pred_indices}'
            f'\n\nSelect Indices'
            f'\n{select_indices}'
            f'\n\nProg Mask'
            f'\n{prog_mask}'
            f'\n\nPred Masks'
            f'\n{pred_masks}'
            f'\n\nSelect Masks'
            f'\n{select_masks}'
            f'\n===================================================================\n'
        )
        return None

    def sample_nonlinearities(self, num_nonlinearities: int):
        """
        Sample non-linearities for each outcome.
        """
        if self.nonlinearity_selection_type == "random":
            # pick num_nonlinearities 
            return random.choices(population=self.nonlinear_fcts, k=num_nonlinearities)
        
        else:
            raise ValueError(f"Unknown nonlinearity selection type {self.selection_type}.")
        

class TSimulator(SimulatorBase):
    """
    Data generation process class for simulating treatment selection only, when counterfactual outcomes are available (as for in-vitro/pharmacoscopy data).
    """
    nonlinear_fcts = [
                lambda x: np.abs(x),
                lambda x: np.exp(-(x**2) / 2),
                lambda x: 1 / (1 + x**2),
                lambda x: np.cos(x),
                lambda x: np.arctan(x),
                lambda x: np.tanh(x),
                lambda x: np.sin(x),
                lambda x: np.log(1 + x**2),
                lambda x: np.sqrt(1 + x**2),
                lambda x: np.cosh(x),
            ]
    
    def __init__(
        self,
        # Data dimensionality
        dim_X: int,

        # Seed
        seed: int = 42,

        # Simulation type
        simulation_type: str = "T",

        # Dimensionality of treatments and outcome
        num_binary_outcome: int = 0,
        standardize_outcome: bool = False,
        standardize_per_outcome: bool = False,
        num_T: int = 3,
        dim_Y: int = 3,

        # Scale parameters
        propensity_scale: float = 1,
        unbalancedness_exp: float = 0,
        nonlinearity_scale: float = 1,
        propensity_type: str = "prog_pred",
        alpha: float = 0.5,
        enforce_balancedness: bool = False,

        # Important features
        num_select_features: int = 5,
        treatment_feature_overlap: bool = False,

        # Feature selection
        random_feature_selection: bool = True,
        nonlinearity_selection_type: bool = True,

        
    ) -> None:
        # Number of features
        self.dim_X = dim_X

        # Make sure results are reproducible by setting seed for np, torch, random
        self.seed = seed
        enable_reproducible_results(seed=self.seed)

        # Simulation type
        self.simulation_type = simulation_type

        # Store dimensions
        self.num_binary_outcome = num_binary_outcome
        self.standardize_outcome = standardize_outcome
        self.standardize_per_outcome = standardize_per_outcome
        self.num_T = num_T
        self.dim_Y = dim_Y

        # Scale parameters
        self.propensity_scale = propensity_scale
        self.unbalancedness_exp = unbalancedness_exp
        self.nonlinearity_scale = nonlinearity_scale
        self.propensity_type = propensity_type
        self.alpha = alpha
        self.enforce_balancedness = enforce_balancedness

        # Important features
        self.num_select_features = num_select_features
        self.treatment_feature_overlap = treatment_feature_overlap
        self.num_important_features = num_select_features

        # Feature selection
        self.random_feature_selection = random_feature_selection
        self.nonlinearity_selection_type = nonlinearity_selection_type

        # Setup variables
        self.nonlinearities = None
        self.select_masks = None
        self.select_weights = None

        # Setup
        self.setup()

        # Simulation variables
        self.X = None
        self.select_scores = None
        self.propensities, self.outcomes, self.T, self.Y = None, None, None, None

    def get_simulated_data(self, train_ratio: float = 0.8):
        """
        Extract results and split into training and test set. Include counterfactual outcomes and propensities.
        """
        return self.X, self.T, self.Y, self.outcomes, self.propensities
        # Split data
        # train_size = int(train_ratio * self.X.shape[0])
        # X_train, X_test = self.X[:train_size], self.X[train_size:]
        # T_train, T_test = self.T[:train_size], self.T[train_size:]
        # Y_train, Y_test = self.Y[:train_size], self.Y[train_size:]
        # outcomes_train, outcomes_test = self.outcomes[:train_size,:,:], self.outcomes[train_size:,:,:]
        # propensities_train, propensities_test = self.propensities[:train_size], self.propensities[train_size:]

        # if train_ratio == 1:
        #     return self.X, self.T, self.Y, self.outcomes, self.propensities
        
        # return X_train, X_test, T_train, T_test, Y_train, Y_test, outcomes_train, outcomes_test, propensities_train, propensities_test

    def simulate(self, X, outcomes=None) -> Tuple:
        """
        Simulate treatment and outcome for a dataset based on the configuration.
        """
        log.debug(
            f'Simulating treatment and outcome for a dataset with:'
            f'\n==================================================================='
            f'\nDim X: {self.dim_X}'
            f'\nDim T: {self.num_T}'
            f'\nDim Y: {self.dim_Y}'
            f'\nPropensity Scale: {self.propensity_scale}'
            f'\nUnbalancedness Exponent: {self.unbalancedness_exp}'
            f'\nNonlinearity Scale: {self.nonlinearity_scale}'
            f'\nNum Select Features: {self.num_select_features}'
            f'\nFeature Overlap: {self.treatment_feature_overlap}'
            f'\nRandom Feature Selection: {self.random_feature_selection}'
            f'\nNonlinearity Selection Type: {self.nonlinearity_selection_type}'
            f'\n===================================================================\n'
        )

        # 1. Store data
        self.X = X

        # 2. Compute scores for prognostic, predictive, and selective features
        self.compute_scores()

        # 3. Retrieve factual and counterfactual outcomes based on the data and the predictive and prognostic scores
        self.outcomes = outcomes
        assert self.outcomes.shape == (self.X.shape[0], self.num_T, self.dim_Y)

        if self.standardize_outcome:
            if self.standardize_per_outcome:
                self.outcomes = zscore(self.outcomes, axis=0) #, axis=None) # add axis=None to make problem easier again
            else:
                self.outcomes = zscore(self.outcomes, axis=None) #, axis=None) # add axis=None to make problem easier again

        log.debug(
            f'\nCheck if outcomes are processed correctly:'
            f'\n==================================================================='
            f'\n\nOutcomes'
            f'\n{self.outcomes}'
            f'\n{self.outcomes.shape}'
            f'\n\nMean Outcomes'
            f'\n{self.outcomes.mean(axis=0)}'
            f'\n\nVariance Outcomes'
            f'\n{self.outcomes.var(axis=0)}'
            f'\n===================================================================\n'
        )

        # 4. Compute propensities based on the data and the selective scores
        self.compute_propensities()

        # 5. Sample treatment assignment based on the propensities
        self.sample_T()

        # 6. Extract the outcome based on the treatment assignment
        self.extract_Y()

        return None
    
    def setup(self) -> None:
        """
        Setup the simulator by defining variables which remain the same across simulations with different samples but the same configuration.
        """
        # 1. Sample nonlinearities used 
        num_nonlinearities = 1 # Same non-linearity for all treatment selection mechanisms
        self.nonlinearities = self.sample_nonlinearities(num_nonlinearities)

        # 2. Set important feature masks - determine which features should be used for treatment selection, outcome prediction
        self.sample_important_feature_masks()

        # 3. Sample weights for features
        self.sample_uniform_weights()

    def get_true_cates(self, 
                       X: np.ndarray, 
                       T: np.ndarray, 
                       outcomes: np.ndarray) -> np.ndarray:
        """
        Compute true CATEs for each treatment based on the data and the outcomes.
        Always use the selected treatment as the base treatment.
        """
        # Compute CATEs for each treatment
        cates = np.zeros((X.shape[0], self.num_T, self.dim_Y))

        for i in range(X.shape[0]):
            for j in range(self.num_T):
                cates[i,j,:] = outcomes[i,j,:] - outcomes[i,int(T[i]),:]

        log.debug(
            f'\nCheck if true CATEs are computed correctly:'
            f'\n==================================================================='
            f'\nOutcomes: {outcomes.shape}'
            f'\n{outcomes}'
            f'\n\nTreatment Assignment: {T.shape}'
            f'\n{T}'
            f'\n\nTrue CATEs: {cates.shape}'
            f'\n{cates}'
            f'\n===================================================================\n'
        )

        return cates
    
    def extract_Y(self) -> None:
        """
        Extract the outcome based on the treatment assignment.
        """
        self.Y = self.outcomes[np.arange(self.X.shape[0]), self.T] 

        log.debug(
            f'\nCheck if outcomes are extracted correctly:'
            f'\n==================================================================='
            f'\nOutcomes'
            f'\n{self.outcomes}'
            f'\n{self.outcomes.shape}'
            f'\n\nTreatment Assignment'
            f'\n{self.T}'
            f'\n{self.T.shape}'
            f'\n\nExtracted Outcomes'
            f'\n{self.Y}'
            f'\n{self.Y.shape}'
            f'\n===================================================================\n'
        )

        return None

    def sample_T(self) -> None:
        """
        Sample treatment assignment based on the propensities.
        """
        # Sample from the resulting categorical distribution per row
        self.T = np.array([np.random.choice([tre for tre in range(self.propensities.shape[1])], p=row) for row in self.propensities])
   
        log.debug(
            f'\nCheck if treatment assignment is sampled correctly:'
            f'\n==================================================================='
            f'\nPropensities'
            f'\n{self.propensities}'
            f'\n{self.propensities.shape}'
            f'\n\nTreatment Assignment'
            f'\n{self.T}'
            f'\n{self.T.shape}'
            f'\n\nUnique Treatment Counts'
            f'\n{np.unique(self.T, return_counts=True)}'
            f'\n===================================================================\n'
        )

        return None
    
    def get_unbalancedness_weights(self, size: int) -> np.ndarray:
        """
        Create weights for introducing unbalancedness for class probabilities.
        """
        # Sample initial distribution of treatment assignment 
        unb_weights = np.random.uniform(0, 1, size=size) 
        unb_weights = unb_weights / unb_weights.sum()

        # Standardize the weights and make sure that a treatment doesn't completely disappear for small unbalancedness exponents
        min_val = unb_weights.min()
        range_val = unb_weights.max() - min_val
        unb_weights = (unb_weights - min_val) / range_val
        unb_weights = 0.01 + unb_weights * 0.98

        return unb_weights
    
    def compute_propensities(self) -> None:
        """
        Compute propensities based on the data and the selective scores.
        """
        select_scores_none = zscore(self.select_scores, axis=0) # Comment for Predictive Epertise

        select_scores_pred = np.zeros((self.X.shape[0], self.num_T))
        select_scores_pred_flipped = np.zeros((self.X.shape[0], self.num_T))
        select_scores_prog = np.zeros((self.X.shape[0], self.num_T))
        select_scores_tre = np.zeros((self.X.shape[0], self.num_T))

        select_scores_pred[:,0] = self.outcomes[:,0,0] - self.outcomes[:,1,0]
        select_scores_pred[:,1] = self.outcomes[:,1,0] - self.outcomes[:,0,0]

        select_scores_pred_flipped[:,0] = self.outcomes[:,1,0] - self.outcomes[:,0,0]
        select_scores_pred_flipped[:,1] = self.outcomes[:,0,0] - self.outcomes[:,1,0]

        select_scores_prog[:,0] = self.outcomes[:,0,0]
        select_scores_prog[:,1] = -self.outcomes[:,0,0]

        select_scores_tre[:,0] = -self.outcomes[:,1,0]
        select_scores_tre[:,1] = self.outcomes[:,1,0]

        if self.propensity_type == "prog_tre":
            scores = self.alpha * select_scores_tre + (1 - self.alpha) * select_scores_prog

        # Standardize all scores
        select_scores_pred = zscore(select_scores_pred, axis=0)
        select_scores_pred_flipped = zscore(select_scores_pred_flipped, axis=0)
        select_scores_prog = zscore(select_scores_prog, axis=0)
        select_scores_tre = zscore(select_scores_tre, axis=0)

        if self.propensity_type == "prog_pred":
            scores = self.alpha * select_scores_pred + (1 - self.alpha) * select_scores_prog

        elif self.propensity_type == "prog_tre":
            pass

        elif self.propensity_type == "none_prog":
            scores = self.alpha * select_scores_prog + (1 - self.alpha) * select_scores_none

        elif self.propensity_type == "none_pred":
            scores = self.alpha * select_scores_pred + (1 - self.alpha) * select_scores_none

        elif self.propensity_type == "none_tre":
            scores = self.alpha * select_scores_tre + (1 - self.alpha) * select_scores_none

        elif self.propensity_type == "none_pred_flipped":
            scores = self.alpha * select_scores_pred_flipped + (1 - self.alpha) * select_scores_none

        elif self.propensity_type == "pred_pred_flipped":
            scores = self.alpha * select_scores_pred_flipped + (1 - self.alpha) * select_scores_pred

        elif self.propensity_type == "rct_none":
            scores = select_scores_none

        else:
            raise ValueError(f"Unknown propensity type {self.propensity_type}.")

        if self.enforce_balancedness:
            scores = zscore(scores, axis=0)

        if self.propensity_type == "rct_none":
            scores = self.alpha * select_scores_none

        # Introduce unbalancedness and manipulate unbalancedness weights for comparable experiments with different seeds
        unb_weights = self.get_unbalancedness_weights(size=scores.shape[1])

        # Apply the softmax function to each row to get probabilities
        p = softmax(self.propensity_scale*scores, axis=1)

        # Scale probabilities to introduce unbalancedness
        p = p * (1 - unb_weights) ** self.unbalancedness_exp

        # Make sure rows add up to one again
        row_sums = p.sum(axis=1, keepdims=True)
        p = p / row_sums
        self.propensities = p

        log.debug(
            f'\nCheck if propensities are computed correctly:'
            f'\n==================================================================='
            f'\nSelect Scores'
            f'\n{self.select_scores}'
            f'\n{self.select_scores.shape}'
            f'\n\nPropensities'
            f'\n{self.propensities}'
            f'\n{self.propensities.shape}'
            f'\n===================================================================\n'
        )

        return None

    def compute_scores(self) -> None:
        """
        Compute scores for prognostic, predictive, and selective features based on the data and the feature weights.
        """
        # Each column of the score matrix corresponds to the score for a specific outcome. Rows correspond to samples.
        select_lin = self.X @ self.select_weights.T

        log.debug(
            f'\nCheck if linear scores are computed correctly for selective features:'
            f'\n==================================================================='
            f'\nself.X'
            f'\n{self.X}'
            f'\n{self.X.shape}'
            f'\n\nSelect Weights'
            f'\n{self.select_weights}'
            f'\n{self.select_weights.shape}'
            f'\n\nSelect Lin'
            f'\n{select_lin}'
            f'\n{select_lin.shape}'
            f'\n===================================================================\n'
        )

        # Introduce non-linearity and get final scores
        select_scores = (1 - self.nonlinearity_scale) * select_lin + self.nonlinearity_scale * self.nonlinearities[0](select_lin)
        self.select_scores = select_scores

        return None
    
    @property
    def weights(self) -> Tuple:
        """
        Return weights for prognostic, predictive, and selective features.
        """
        return None, None, self.select_weights
    
    def sample_uniform_weights(self) -> None:
        """
        sample uniform weights for the features.
        """
        # Sample weights for selective features, a weight for every dimension for every treatment and outcome
        select_weights = np.random.uniform(-1, 1, size=(self.num_T, self.dim_X))


        # Mask weights for features that are not important
        for i in range(self.num_T):
            select_weights[i] = select_weights[i] * self.select_masks[:,i]

        log.debug(
            f'\nCheck if masks are applied correctly:'
            f'\n==================================================================='
            f'\nSelect Weights'
            f'\n{select_weights}'
            f'\n{select_weights.shape}'
            f'\n\nSelect Masks'
            f'\n{self.select_masks}'
            f'\n{self.select_masks.shape}'
            f'\n===================================================================\n'
        )
           
        self.select_weights = select_weights

        return None
    @property
    def all_important_features(self) -> np.ndarray:
        """
        Return all important feature indices.
        """
        all_important_features = self.selective_features
        log.debug(
            f'\nCheck if all important features are computed correctly:'
            f'\n==================================================================='
            f'\n\nSelect Features'
            f'\n{self.selective_features}'
            f'\n\nAll Important Features'
            f'\n{all_important_features}'
            f'\n===================================================================\n'
        )

        return all_important_features
    
    @property
    def predictive_features(self) -> np.ndarray:
        """
        Return predictive feature indices.
        """
        return None
    
    @property
    def prognostic_features(self) -> np.ndarray:
        """
        Return prognostic feature indices.
        """
        return None

    @property
    def selective_features(self) -> np.ndarray:
        """
        Return selective feature indices.
        """
        select_features = np.where((self.select_masks.sum(axis=1)).astype(np.int32) != 0)
        return select_features
    
    def sample_important_feature_masks(self) -> None:
        """
        Pick features that are important for treatment selection based on the configuration.
        """
        # Get indices for features and shuffle if random_feature_selection is True
        all_indices = np.arange(self.dim_X)

        if self.random_feature_selection:
            np.random.shuffle(all_indices)

        # Initialize masks
        select_masks = np.zeros(shape=(self.dim_X, self.num_T))

        # Handle case with feature overlap
        if self.treatment_feature_overlap:
            assert self.num_select_features <= int(self.dim_X)
            select_indices = np.array(self.num_T * [all_indices[:self.num_select_features]])
            select_masks[select_indices] = 1

        # Handle case without feature overlap
        else:
            assert (self.num_T * self.num_select_features) <= int(self.dim_X)
            select_indices = all_indices[:self.num_select_features*self.num_T]
            
            # Mask features for every treatment
            for i in range(self.num_T):
                select_masks[select_indices[i*self.num_select_features:(i+1)*self.num_select_features],i] = 1

        self.select_masks = select_masks

        return None

    def sample_nonlinearities(self, num_nonlinearities: int):
        """
        Sample non-linearities for each outcome.
        """
        if self.nonlinearity_selection_type == "random":
            # pick num_nonlinearities 
            return random.choices(population=self.nonlinear_fcts, k=num_nonlinearities)
        
        else:
            raise ValueError(f"Unknown nonlinearity selection type {self.selection_type}.")
