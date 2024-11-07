from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import pandas as pd
#pd.set_option('future.no_silent_downcasting', True) # for future compatibility of .replace method

# Define abstract classes

class TreatmentMechanism(ABC):
    """
    Represents general treatment selection mechanism implementations. Each mechanism should provide an evaluate
    method. 
    """

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, var_names: dict) -> np.ndarray:
        raise NotImplementedError

class OutcomeMechanism(ABC):
    """
    Represents general outcome mechanism implementations. Each mechanism should provide an evaluate
    method. 
    """

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, treatment: np.ndarray, var_names: dict) -> np.ndarray:
        raise NotImplementedError
    
# Define Specialized mechnisms inheriting from classes above

class MelanomaTreatmentMechanism(TreatmentMechanism):
    """
    Linear mapping for Melanoma test dataset.
    """
    def __init__(self, alpha = 2):
        self.alpha = alpha

    def evaluate(self, df: pd.DataFrame, var_names: dict = None) -> np.ndarray:
        # check availability of data
        if not set(["03_SMARCA4", "03_IDO2", "03_FGF19"]).issubset(set(df.columns)):
            raise Exception("Data does not contain features used for defining mechanisms")

        # load data
        SMARCA4 = df["03_SMARCA4"]
        IDO2 = df["03_IDO2"]
        FGF19 = df["03_FGF19"]

        # get sum
        sum = SMARCA4 + IDO2 + FGF19 * 0.5
        
        # define treatment from that
        treatment = sum < 1

        return treatment

class MelanomaOutcomeMechanism(OutcomeMechanism):
    """
    Linear mapping for Melanoma test dataset.
    """
    def __init__(self, confounding: bool = False):
        self.confounding = confounding

    def evaluate(self, df: pd.DataFrame, treatment: np.ndarray, var_names: dict = None) -> np.ndarray:
        # check availability of data
        if not set(["03_SMARCA4", "03_IDO2", "03_FGF19"]).issubset(set(df.columns)):
            raise Exception("Data does not contain features used for defining mechanisms")

        # load data
        SMARCA4 = df["03_SMARCA4"]
        IDO2 = df["03_IDO2"]
        FGF19 = df["03_FGF19"]

        # define outcome
        outcome = treatment * (SMARCA4 + IDO2 + 2) + FGF19

        return outcome

class AdditiveNoiseMechanism(TreatmentMechanism):
    """
    Some mechanism of the form f(x) + noise.
    """
    def __init__(self, predictor, noise_std):
        self.predictor = predictor
        self.noise_std = noise_std

    def evaluate(self, X: np.ndarray, var_names: dict = None) -> np.ndarray:
        noise_dim = X.shape[0]
        noise_vec = multivariate_normal.rvs(mean=np.zeros(noise_dim), cov=np.eye(noise_dim) * self.noise_std)
        return self.predictor.evaluate(X) + noise_vec
    
class Generator:
    """
    Instances of this class store data for features, controls, instruments and confounders and 
    generate treatents and outcomes based on the data and user-defined mechanisms. 
    """
    def __init__(self, df, treatment, outcome, features, controls, instruments):
        # Variable storing dataframe
        self.df = df
        
        # Variable names and causal structure in CATE setting
        self.var_names = {
            "treatment": treatment,
            "outcome": outcome,
            "features": features,
            "controls": controls,
            "instruments": instruments
        }
        
        # Mechanisms
        self._treatment_mechanism = None
        self._outcome_mechanism = None
    

    def generate_factual(self) -> pd.DataFrame():
        # Generate treatment and outcome sequentially
        treatment = self._treatment_mechanism.evaluate(self.df)
        outcome = self._outcome_mechanism.evaluate(self.df, treatment)

        # Store results in dataframe
        result_df = self.df.copy()
        result_df[self.var_names["treatment"]] = treatment
        result_df[self.var_names["outcome"]] = outcome

        return result_df
    
    def generate_all(self, treatment_values) -> pd.DataFrame():
        # result df
        result_df = pd.DataFrame()

        # Iterate through all possible values 
        n = len(self.df)
        for i, value in enumerate(sorted(treatment_values)):
            treatment = np.full(n, value)
            outcome = self._outcome_mechanism.evaluate(self.df, treatment)
            colname = 'y0_a'+ str(i) + '_' + str(value)
            result_df[colname] = outcome

        return result_df

    @property
    def treatment_mechanism(self):
        return self._treatment_mechanism

    @property
    def outcome_mechanism(self):
        return self._outcome_mechanism

    @treatment_mechanism.setter
    def treatment_mechanism(self, new_treatment_mechanism):
        if isinstance(new_treatment_mechanism, TreatmentMechanism):
            self._treatment_mechanism = new_treatment_mechanism
        else:
            raise TypeError("Only mechanisms which inherit from the abstract class TreatmentMechanism are allowed.")
    
    @outcome_mechanism.setter
    def outcome_mechanism(self, new_outcome_mechanism):
        if isinstance(new_outcome_mechanism, OutcomeMechanism):
            self._outcome_mechanism = new_outcome_mechanism
        else:
            raise TypeError("Only mechanisms which inherit from the abstract class OutcomeMechanism are allowed.")