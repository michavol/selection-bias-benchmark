from pathlib import Path
import os
import catenets.models as cate_models
import numpy as np
import pandas as pd

import src.iterpretability.logger as log
from src.iterpretability.explain import Explainer
from src.iterpretability.datasets.data_loader import load
from src.iterpretability.synthetic_simulate import (
    SyntheticSimulatorLinear,
    SyntheticSimulatorModulatedNonLinear,
)
from src.iterpretability.utils import (
    attribution_accuracy,
    compute_pehe,
)

# For contour plotting
import umap 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import imageio
import torch
import shap

def get_learners(model_list, X_train, Y_train, n_iter, batch_size, batch_norm, discrete_treatment=True, discrete_outcome=False):
    learners = {
                    "TLearner": cate_models.torch.TLearner(
                        X_train.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=batch_size,
                        n_iter=n_iter,
                        batch_norm=batch_norm,
                        nonlin="relu",
                    ),
                    "SLearner": cate_models.torch.SLearner(
                        X_train.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        n_iter=n_iter,
                        batch_size=batch_size,
                        batch_norm=batch_norm,
                        nonlin="relu",
                    ),
                    "TARNet": cate_models.torch.TARNet(
                        X_train.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_r=1,
                        n_layers_out=1,
                        n_units_out=100,
                        n_units_r=100,
                        batch_size=batch_size,
                        n_iter=n_iter,
                        batch_norm=batch_norm,
                        nonlin="relu",
                    ),
                    "DRLearner": cate_models.torch.DRLearner(
                        X_train.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        n_iter=n_iter,
                        batch_size=batch_size,
                        batch_norm=batch_norm,
                        nonlin="relu",
                    ),
                    "XLearner": cate_models.torch.XLearner(
                        X_train.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        n_iter=n_iter,
                        batch_size=batch_size,
                        batch_norm=batch_norm,
                        nonlin="relu",
                    ),
                    "CFRNet_0.01": cate_models.torch.TARNet(
                        X_train.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_r=1,
                        n_layers_out=1,
                        n_units_out=100,
                        n_units_r=100,
                        batch_size=batch_size,
                        n_iter=n_iter,
                        batch_norm=batch_norm,
                        nonlin="relu",
                        penalty_disc=0.01,
                    ),
                    "CFRNet_0.001": cate_models.torch.TARNet(
                        X_train.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_r=1,
                        n_layers_out=1,
                        n_units_out=100,
                        n_units_r=100,
                        batch_size=batch_size,
                        n_iter=n_iter,
                        batch_norm=batch_norm,
                        nonlin="relu",
                        penalty_disc=0.001,
                    ),
                    "CFRNet_0.0001": cate_models.torch.TARNet(
                        X_train.shape[1],
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_r=1,
                        n_layers_out=1,
                        n_units_out=100,
                        n_units_r=100,
                        batch_size=batch_size,
                        n_iter=n_iter,
                        batch_norm=batch_norm,
                        nonlin="relu",
                        penalty_disc=0.0001,
                    ),
                    "EconML_CausalForestDML": cate_models.econml.EconMlEstimator(
                        model_name="EconML_CausalForestDML",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),
                    "EconML_DMLOrthoForest": cate_models.econml.EconMlEstimator(
                        model_name="EconML_DMLOrthoForest",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),

                    "EconML_SparseLinearDML": cate_models.econml.EconMlEstimator(
                        model_name="EconML_SparseLinearDML",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),
                    "EconML_SparseLinearDRLearner": cate_models.econml.EconMlEstimator(
                        model_name="EconML_SparseLinearDRLearner",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),
                    "EconML_LinearDRLearner": cate_models.econml.EconMlEstimator(
                        model_name="EconML_LinearDRLearner",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),
                    "EconML_DRLearner": cate_models.econml.EconMlEstimator(
                        model_name="EconML_DRLearner",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),
                    "EconML_XLearner": cate_models.econml.EconMlEstimator(
                        model_name="EconML_XLearner",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),
                    "EconML_SLearner": cate_models.econml.EconMlEstimator(
                        model_name="EconML_SLearner",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),
                    "EconML_TLearner": cate_models.econml.EconMlEstimator(
                        model_name="EconML_TLearner",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),
                    "EconML_SparseLinearDRIV": cate_models.econml.EconMlEstimator(
                        model_name="EconML_SparseLinearDRIV",
                        # cv_model_selection=3,
                        discrete_treatment=discrete_treatment,
                        discrete_outcome=discrete_outcome,
                    ),

                }
    
    for name in model_list:
        if name not in learners:
            raise Exception(f"Unknown model name {name}.")
        
    # Only return the learners that are in the model_list
    learners = {name: learners[name] for name in model_list}
    
    return learners

def get_learner_explanations(learners, X_test, X_train, Y_train, W_train, explainer_limit, explainer_list, return_learners=False, already_trained=False):
    learner_explainers = {}
    learner_explanations = {}

    for name in learners:
        log.info(f"Fitting {name}.")

        if not already_trained:
            learners[name].fit(X=X_train, y=Y_train, w=W_train)
        
        log.info(f"Explaining {name}.")

        if "EconML" in name:
            shap_values = learners[name].est.shap_values(X_test[:explainer_limit], background_samples=None)
            treatment_names = learners[name].est.cate_treatment_names()
            output_names = learners[name].est.cate_output_names()
            output_name = output_names[0]
            treatment_name = treatment_names[0]
            learner_explanations[name] = {"kernel_shap" : shap_values[output_name][treatment_name].values} 
            
        else:
            learner_explainers[name] = Explainer(
                learners[name],
                feature_names=list(range(X_train.shape[1])),
                explainer_list=explainer_list,
            )
            learner_explanations[name] = learner_explainers[name].explain(
                X_test[: explainer_limit]
            )

    if return_learners:
        return learner_explanations, learners
    else:
        return learner_explanations

class PredictiveSensitivity:
    """
    Sensitivity analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        batch_norm: bool = False,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        propensity_type: str = "pred",
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5, 1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
        selection_type: str = "random",
        non_linearity_scale: float = 0,
        model_list: list = ["TLearner"]
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.propensity_type = propensity_type
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type
        self.selection_type = selection_type
        self.non_linearity_scale = non_linearity_scale
        self.model_list = model_list

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
        ],
        debug: bool = False,
        directory_path_: str = None,
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio, debug=debug, directory_path_=directory_path_)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        elif self.synthetic_simulator_type == "nonlinear":
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=self.non_linearity_scale,
                seed=self.seed,
                selection_type=self.selection_type,
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
            )

            log.info("Fitting and explaining learners...")

            learners = get_learners(
                model_list=self.model_list,
                X_train=X_train,
                Y_train=Y_train,
                n_iter=self.n_iter,
                batch_size=self.batch_size,
                batch_norm=self.batch_norm,
                discrete_outcome=binary_outcome
            )

            learner_explanations = get_learner_explanations(learners, 
                                                            X_test, X_train, Y_train, W_train, 
                                                            self.explainer_limit, explainer_list)

            all_important_features = sim.get_all_important_features(with_selective=True)
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()
            select_features = sim.get_selective_features()

            cate_test = sim.te(X_test)

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    acc_scores_selective_features = attribution_accuracy(
                        select_features, attribution_est
                    )
                    cate_pred = learners[learner_name].predict(X=X_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_predictive_features,
                            acc_scores_prog_features,
                            acc_scores_selective_features,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "Pred features ACC",
                "Prog features ACC",
                "Select features ACC",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = self.save_path / "results/predictive_sensitivity"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )


class NonLinearitySensitivity:
    """
    Sensitivity analysis for nonlinearity in prognostic and predictive functions.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        batch_norm: bool = False,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        propensity_type: str = "pred",
        nonlinearity_scales: list = [0.0, 0.2, 0.5, 0.7, 1.0],
        selection_type: str = "random",
        predictive_scale: float = 1,
        synthetic_simulator_type: str = "random",
        model_list: list = ["TLearner"]
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.propensity_type = propensity_type
        self.nonlinearity_scales = nonlinearity_scales
        self.selection_type = selection_type
        self.predictive_scale = predictive_scale
        self.synthetic_simulator_type = synthetic_simulator_type
        self.model_list = model_list

    def run(
        self,
        dataset: str = "tcga_100",
        num_important_features: int = 15,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
        ],
        train_ratio: float = 0.8,
        binary_outcome: bool = False,
        debug=False,
        directory_path_: str = None,

    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )
        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio, debug=debug, directory_path_=directory_path_)
        explainability_data = []

        for nonlinearity_scale in self.nonlinearity_scales:
            log.info(f"Now working with a nonlinearity scale {nonlinearity_scale}...")

            if self.synthetic_simulator_type == "linear":
                raise Exception("Linear simulator not supported for nonlinearity sensitivity.")
            
            elif self.synthetic_simulator_type == "nonlinear":
                sim = SyntheticSimulatorModulatedNonLinear(
                    X_raw_train,
                    num_important_features=num_important_features,
                    non_linearity_scale=nonlinearity_scale,
                    seed=self.seed,
                    selection_type=self.selection_type
                )
            else:
                raise Exception("Unknown simulator type.")
            
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
            )
            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
            )

            log.info("Fitting and explaining learners...")
            learners = get_learners(
                model_list=self.model_list,
                X_train=X_train,
                Y_train=Y_train,
                n_iter=self.n_iter,
                batch_size=self.batch_size,
                batch_norm=False,
                discrete_outcome=binary_outcome
            )

            learner_explanations = get_learner_explanations(learners, 
                                                            X_test, X_train, Y_train, W_train, 
                                                            self.explainer_limit, explainer_list)

            all_important_features = sim.get_all_important_features(with_selective=True)
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()
            select_features = sim.get_selective_features()

            cate_test = sim.te(X_test)

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    acc_scores_selective_features = attribution_accuracy(
                        select_features, attribution_est
                    )
                    

                    cate_pred = learners[learner_name].predict(X=X_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            nonlinearity_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_predictive_features,
                            acc_scores_prog_features,
                            acc_scores_selective_features,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Nonlinearity Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "Pred features ACC",
                "Prog features ACC",
                "Select features ACC",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / f"results/nonlinearity_sensitivity/{self.synthetic_simulator_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}-seed{self.seed}.csv"
        )


class PropensitySensitivity:
    """
    Sensitivity analysis for confounding.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        batch_norm: bool = False,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
        nonlinearity_scale: float = 0,
        selection_type: str = "random",
        propensity_type: str = "pred",
        propensity_scales: list = [0, 0.5, 1, 2, 5, 10],
        model_list: list = ["TLearner"]
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type
        self.nonlinearity_scale = nonlinearity_scale
        self.selection_type = selection_type
        self.propensity_type = propensity_type
        self.propensity_scales = propensity_scales
        self.model_list = model_list

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        predictive_scale: float = 1,
        nonlinearity_scale: float = 0.5,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
        ],
        debug: bool = False,
        directory_path_: str = None,
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features} and predictive scale {predictive_scale}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio, debug=debug, directory_path_=directory_path_)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        elif self.synthetic_simulator_type == "nonlinear":
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=self.nonlinearity_scale,
                seed=self.seed,
                selection_type=self.selection_type,
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []

        for propensity_scale in self.propensity_scales:
            log.info(f"Now working with propensity_scale = {propensity_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            
            log.info("Fitting and explaining learners...")
            learners = get_learners(
                model_list=self.model_list,
                X_train=X_train,
                Y_train=Y_train,
                n_iter=self.n_iter,
                batch_size=self.batch_size,
                batch_norm=self.batch_norm,
                discrete_outcome=binary_outcome
            )

            learner_explanations = get_learner_explanations(learners, 
                                                            X_test, X_train, Y_train, W_train, 
                                                            self.explainer_limit, explainer_list)

            all_important_features = sim.get_all_important_features(with_selective=False)
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features = attribution_accuracy(
                        prog_features, attribution_est
                    )

                    cate_pred = learners[learner_name].predict(X=X_test)
                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            propensity_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_predictive_features,
                            acc_scores_prog_features,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Propensity Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "Pred features ACC",
                "Prog features ACC",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / f"results/propensity_sensitivity/{self.synthetic_simulator_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"propensity_scale_{dataset}_{num_important_features}_"
            f"proptype_{self.propensity_type}_"
            f"predscl_{predictive_scale}_"
            f"nonlinscl_{nonlinearity_scale}_"
            f"trainratio_{train_ratio}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )


class CohortSizeSensitivity:
    """
    Sensitivity analysis for varying numbers of samples. This experiment will generate a .csv with the recorded metrics.
    It will also generate a gif, showing the progression on dimensionality-reduced spaces.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        batch_norm: bool = False,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        propensity_type: str = "selective",
        cohort_sizes: list = [0.5, 0.7, 1.0],
        nonlinearity_scale: float = 0.5,
        predictive_scale: float = 1,
        synthetic_simulator_type: str = "random",
        selection_type: str = "random",
        model_list: list = ["TLearner"],
        num_cube_samples: int = 1000,
        dim_reduction_method: str = "umap",
        dim_reduction_on_important_features: bool = True,
        visualize_progression: bool = True,
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.propensity_type = propensity_type
        self.cohort_sizes = cohort_sizes
        self.nonlinearity_scale = nonlinearity_scale
        self.predictive_scale = predictive_scale
        self.synthetic_simulator_type = synthetic_simulator_type
        self.selection_type = selection_type
        self.model_list = model_list
        self.num_cube_samples = num_cube_samples
        self.dim_reduction_method = dim_reduction_method
        self.dim_reduction_on_important_features = dim_reduction_on_important_features
        self.visualize_progression = visualize_progression

    def run(
        self,
        dataset: str = "tcga_100",
        num_important_features: int = 15,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
        ],
        train_ratio: float = 0.8,
        binary_outcome: bool = False,
        debug=False,
        directory_path_: str = None,

    ) -> None:
        # Log setting
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        # Load data
        X_raw_train_full, X_raw_test_full = load(dataset, train_ratio=train_ratio, debug=debug, directory_path_=directory_path_)
        explainability_data = []

        # Simulate treatment and outcome for train and test
        sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train_full,
                num_important_features=num_important_features,
                non_linearity_scale=self.nonlinearity_scale,
                seed=self.seed,
                selection_type=self.selection_type
            )
        
        (
            X_train_full,
            W_train_full,
            Y_train_full,
            po0_train_full,
            po1_train_full,
            propensity_train_full
        ) = sim.simulate_dataset(
                X_raw_train_full,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
            )
        
        X_test_full, W_test_full, Y_test_full, po0_test_full, po1_test_full, _ = sim.simulate_dataset(
            X_raw_test_full,
            predictive_scale=self.predictive_scale,
            binary_outcome=binary_outcome,
            treatment_assign=self.propensity_type,
        )

        # Retrieve important features
        all_important_features = sim.get_all_important_features(with_selective=True)
        pred_features = sim.get_predictive_features()
        prog_features = sim.get_prognostic_features()
        select_features = sim.get_selective_features()

        # Code for sampling from hypercube -> does not work well because data lives in very small part of that hypercube
        # # Sample from a hypercube grid for the X_raw_train_full dataset - making a complete hypercube grid would be too large
        # # Sample for important features only as we know these are the ones that matter and will make sampling from a hypercube more meaningful
        # X_raw_train_full_important = X_raw_train_full[:, all_important_features]
        # min_vals = X_raw_train_full_important.min(axis=0)
        # max_vals = X_raw_train_full_important.max(axis=0)

        # # Add some relative padding to the min and max values
        # min_vals = min_vals - 0.1 * np.abs(min_vals)
        # max_vals = max_vals + 0.1 * np.abs(max_vals)

        # # Sample points
        # grid_samples = np.zeros((self.num_cube_samples, X_raw_train_full.shape[1]))
        # grid_samples[:, all_important_features] = np.random.uniform(min_vals, max_vals, (self.num_cube_samples, X_raw_train_full_important.shape[1]))

        if self.visualize_progression:
            # Use full training set with added noisy samples as focused samples of space
            std_dev = np.std(X_raw_train_full, axis=0)
            grid_samples = X_raw_train_full
            grid_samples = np.vstack([grid_samples,
                                        X_raw_train_full + std_dev*np.random.normal(0, 1, X_raw_train_full.shape),
                                        X_raw_train_full + 0.1*std_dev*np.random.normal(0, 1, X_raw_train_full.shape),
                                        X_raw_train_full + 0.1*std_dev*np.random.normal(0, 1, X_raw_train_full.shape),
                                        # X_raw_train_full + 0.3*std_dev*np.random.normal(0, 1, X_raw_train_full.shape),
                                        X_raw_train_full + 0.1*std_dev*np.random.normal(0, 1, X_raw_train_full.shape),])

            # Reduce samples to two dimensions for plotting using umap
            if self.dim_reduction_method == "umap":
                reducer = umap.UMAP(min_dist=1, n_neighbors=30, spread=1)
                reducer_shap = umap.UMAP(min_dist=3, n_neighbors=40, spread=4)
                reducer_shap_prop = umap.UMAP(min_dist=2, n_neighbors=30, spread=3)

            elif self.dim_reduction_method == "pca":
                reducer = PCA(n_components=2)
                reducer_shap = PCA(n_components=2)
                reducer_shap_prop = PCA(n_components=2)

            elif self.dim_reduction_method == "tsne":
                raise Exception("t-SNE not supported for this analysis. Does not offer .transform() method.")

            else:
                raise Exception("Unknown dimensionality reduction method.")

            # Fit on grid samples and training data
            if self.dim_reduction_on_important_features:
                grid_samples_2d = reducer.fit_transform(grid_samples[:, all_important_features])
                train_samples_2d = reducer.transform(X_raw_train_full[:, all_important_features])
            else:
                grid_samples_2d = reducer.fit_transform(grid_samples)
                train_samples_2d = reducer.transform(X_raw_train_full)

            # Get model learners (here only one) and explanations for grid samples
            learners = get_learners(
                    model_list=self.model_list,
                    X_train=X_train_full,
                    Y_train=Y_train_full,
                    n_iter=self.n_iter,
                    batch_size=self.batch_size,
                    batch_norm=False,
                    discrete_outcome=binary_outcome
                )
        
            learner_explanations, learners = get_learner_explanations(learners, 
                                                                grid_samples, X_train_full, Y_train_full, W_train_full, 
                                                                grid_samples.shape[0], explainer_list,
                                                                return_learners=True)
            
            learner_explanations_train = get_learner_explanations(learners,
                                                                X_train_full, X_train_full, Y_train_full, W_train_full,
                                                                X_train_full.shape[0], explainer_list,
                                                                already_trained=True)
            
            # Get shap for grid samples
            shap_values_grid = learner_explanations[self.model_list[0]][explainer_list[0]]
            shap_values_train = learner_explanations_train[self.model_list[0]][explainer_list[0]]

            # Perform logistic regression for propensity
            est_prop = LogisticRegression().fit(X_train_full, W_train_full)
            explainer_prop = shap.LinearExplainer(est_prop, X_train_full)
            shap_values_prop_grid = explainer_prop.shap_values(grid_samples)
            shap_values_prop_train = explainer_prop.shap_values(X_train_full)
            
            # Fit umap reducer for shap grid samples 
            if self.dim_reduction_on_important_features:
                shap_values_grid_2d = reducer_shap.fit_transform(shap_values_grid[:, all_important_features])
                shap_values_train_2d = reducer_shap.transform(shap_values_train[:, all_important_features])
                shap_values_prop_grid_2d = reducer_shap_prop.fit_transform(shap_values_prop_grid[:, all_important_features])
                shap_values_prop_train_2d = reducer_shap_prop.transform(shap_values_prop_train[:, all_important_features])
            else:
                shap_values_grid_2d = reducer_shap.fit_transform(shap_values_grid)
                shap_values_train_2d = reducer_shap.transform(shap_values_train)
                shap_values_prop_grid_2d = reducer_shap_prop.fit_transform(shap_values_prop_grid)
                shap_values_prop_train_2d = reducer_shap_prop.transform(shap_values_prop_train)

            # Initialize variable for storing frames
            frames = []  # To store each frame for the GIF
        
        cohort_size_full = X_train_full.shape[0]
        for cohort_size_perc in self.cohort_sizes:
            cohort_size = int(cohort_size_perc * cohort_size_full)

            # Get a subset of the training data
            X_train = X_train_full[:cohort_size]
            W_train = W_train_full[:cohort_size]
            Y_train = Y_train_full[:cohort_size]
            po0_train = po0_train_full[:cohort_size]
            po1_train = po1_train_full[:cohort_size]
            propensity_train = propensity_train_full[:cohort_size]
            cohort_size_train = X_train.shape[0]

            # Get subsets for test data
            X_test = X_test_full[:cohort_size]
            W_test = W_test_full[:cohort_size]
            Y_test = Y_test_full[:cohort_size]
            po0_test = po0_test_full[:cohort_size]
            po1_test = po1_test_full[:cohort_size]

            log.info(f"Now working with a cohort size of {cohort_size}/{X_train_full.shape[0]}...")
            log.info("Fitting and explaining learners...")
            learners = get_learners(
                model_list=self.model_list,
                X_train=X_train,
                Y_train=Y_train,
                n_iter=self.n_iter,
                batch_size=self.batch_size,
                batch_norm=False,
                discrete_outcome=binary_outcome
            )

            # Get learners and explanations for training data
            learner_explanations, learners = get_learner_explanations(learners, 
                                                            X_train, X_train, Y_train, W_train, 
                                                            X_train.shape[0], explainer_list,
                                                            return_learners=True)

            if self.visualize_progression:
                # Make logistic regression for propensity
                est_prop = LogisticRegression().fit(X_train, W_train)
                explainer_prop = shap.LinearExplainer(est_prop, X_train)

                # Set up plot
                fig, axs = plt.subplots(2, 4, figsize=(15, 10))
                eff_grid = sim.te(grid_samples)
                prop_grid = sim.prop(grid_samples)

                # Get cate estimator
                est_eff = learners[self.model_list[0]]
                cate_pred_train = est_eff.predict(X=X_train)

                # Get predictions for the first model
                p_eff_grid = est_eff.predict(X=grid_samples)
                p_prop_grid = est_prop.predict_proba(grid_samples)[:, 1]
                outcomes = [p_prop_grid, prop_grid, p_eff_grid, eff_grid]
                titles = ['prop(x)', 'prop_true(x)', 'cate(x)', 'cate_true(x)']

                # # Get X_Train in 2d
                # shap_values_train = learner_explanations[self.model_list[0]][explainer_list[0]]
                # shap_values_prop_train = explainer_prop.shap_values(X_train)

                # if self.dim_reduction_on_important_features:
                #     X_train_2d = reducer.transform(X_train[:, all_important_features])
                # else:
                #     X_train_2d = reducer.transform(X_train)

                # # Get shap values of X_Train in 2d
                # if self.dim_reduction_on_important_features:
                #     shap_values_train_2d = reducer_shap.transform(shap_values_train[:, all_important_features])
                #     shap_values_prop_train_2d = reducer_shap_prop.transform(shap_values_prop_train[:, all_important_features])
                # else:
                #     shap_values_train_2d = reducer_shap.transform(shap_values_train)
                #     shap_values_prop_train_2d = reducer_shap_prop.transform(shap_values_prop_train)


                # If there are any non-finite elements in the outcomes, remove them from the grid_samples and the other outcomes and raise a warning
                for i, outcome in enumerate(outcomes):
                    if type(outcome) == torch.Tensor:
                        outcome = outcome.cpu().detach().numpy()

                    outcome = np.array(outcome)
                    if not np.all(np.isfinite(outcome)):
                        print("-----------")
                        print(i, np.sum(~np.isfinite(outcome)))
                        log.warning(f'Found non-finite elements in outcomes. Removing {np.sum(~np.isfinite(outcome))} elements.')
                        mask = np.isfinite(outcome)
                        grid_samples_2d = grid_samples_2d[mask]
                        for j in range(len(outcomes)):
                            outcomes[j] = outcomes[j][mask]
                        break

                for j, outcome in enumerate(outcomes):
                    if type(outcome) == torch.Tensor:
                        outcome = outcome.cpu().detach().numpy()

                    # Plot settings
                    cmap = "viridis"
                    s = 15 # 4 for many samples
                    alpha = None
                    edgecolors = "w"
                    linewidths = 0.2
                    
                    # Plot contours
                    if j == 0 or j == 1:
                        tcf = axs[0][j].tricontourf(grid_samples_2d[:,0], 
                                                    grid_samples_2d[:,1], 
                                                    outcome.ravel(), 15, cmap=cmap, levels=50)
                        
                        tcf_shap = axs[1][j].tricontourf(shap_values_prop_grid_2d[:,0], 
                                                        shap_values_prop_grid_2d[:,1], 
                                                        outcome.ravel(), 15, cmap=cmap, levels=50)
                    else:
                        tcf = axs[0][j].tricontourf(grid_samples_2d[:,0], 
                                                    grid_samples_2d[:,1], 
                                                    outcome.ravel(), 15, cmap=cmap, levels=50)
                        
                        tcf_shap = axs[1][j].tricontourf(shap_values_grid_2d[:,0], 
                                                        shap_values_grid_2d[:,1], 
                                                        outcome.ravel(), 15, cmap=cmap, levels=50)

                    
                    # Version: Plot in shape space from current model
                    
                    # if j == 0:
                    #     axs[0][j].scatter(X_train_2d[:,0], X_train_2d[:,1], c=W_train, cmap='coolwarm', edgecolors=edgecolors, s=s, label='Training data for a', alpha=alpha)
                    #     axs[1][j].scatter(shap_values_prop_train_2d[:,0], shap_values_prop_train_2d[:,1], c=W_train, cmap='coolwarm', edgecolors=edgecolors, s=s, alpha=alpha)
                    #     #fig.colorbar(tcf)
                    # if j == 2:
                    #     axs[0][j].scatter(X_train_2d[:,0], X_train_2d[:,1], c=cate_pred_train, cmap='coolwarm', edgecolors=edgecolors, s=s, label='Training data for a', alpha=alpha)
                    #     axs[1][j].scatter(shap_values_train_2d[:,0], shap_values_train_2d[:,1], c=Y_train, cmap='coolwarm', edgecolors=edgecolors, s=s, alpha=alpha)
                    #     #fig.colorbar(tcf_shap)

                    # Version: Always plot in same shap space from full model

                    if j == 0:
                        axs[0][j].scatter(train_samples_2d[:cohort_size_train,0], train_samples_2d[:cohort_size_train,1], 
                                        c=W_train, cmap=cmap, s=s, edgecolors=edgecolors, linewidths=linewidths, alpha=0.5)
                        
                        axs[1][j].scatter(shap_values_prop_train_2d[:cohort_size_train,0], shap_values_prop_train_2d[:cohort_size_train,1], 
                                        c=W_train, cmap=cmap, s=s, edgecolors=edgecolors, linewidths=linewidths, alpha=0.5)
                        #fig.colorbar(tcf)
                    if j == 2:
                        axs[0][j].scatter(train_samples_2d[:cohort_size_train,0], train_samples_2d[:cohort_size_train,1], 
                                        c=cate_pred_train, cmap=cmap, s=s, edgecolors=edgecolors, linewidths=linewidths)
                        
                        axs[1][j].scatter(shap_values_train_2d[:cohort_size_train,0], shap_values_train_2d[:cohort_size_train,1], 
                                        c=cate_pred_train, cmap=cmap, s=s, edgecolors=edgecolors, linewidths=linewidths)
                        #fig.colorbar(tcf_shap)


                    # axs[0][j].set_title(f'{titles[j]} (N={cohort_size})_data_space')
                    # axs[0][j].set_xlim([grid_samples_2d[:,0].min(), grid_samples_2d[:,0].max()])
                    # axs[0][j].set_ylim([grid_samples_2d[:,1].min(), grid_samples_2d[:,1].max()])
                    # axs[0][j].legend()

                    # if j == 0 or j == 1:
                    #     axs[1][j].set_title(f'{titles[j]} (N={cohort_size})_shap_prop_space')
                    #     axs[1][j].set_xlim([shap_values_prop_grid_2d[:,0].min(), shap_values_prop_grid_2d[:,0].max()])
                    #     axs[1][j].set_ylim([shap_values_prop_grid_2d[:,1].min(), shap_values_prop_grid_2d[:,1].max()])
                    #     axs[1][j].legend()
                    # else: 
                    #     axs[1][j].set_title(f'{titles[j]} (N={cohort_size})_shap_space')
                    #     axs[1][j].set_xlim([shap_values_grid_2d[:,0].min(), shap_values_grid_2d[:,0].max()])
                    #     axs[1][j].set_ylim([shap_values_grid_2d[:,1].min(), shap_values_grid_2d[:,1].max()])
                    #     axs[1][j].legend()

                # Save the plot to a buffer
                plt.tight_layout()
                plt.savefig('temp_plot.png')
                plt.close()
                frames.append(imageio.imread('temp_plot.png'))

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    acc_scores_selective_features = attribution_accuracy(
                        select_features, attribution_est
                    )
                    

                    cate_pred = learners[learner_name].predict(X=X_test)
                    cate_test = sim.te(X_test)
                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            cohort_size,
                            cohort_size_perc,
                            self.nonlinearity_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_predictive_features,
                            acc_scores_prog_features,
                            acc_scores_selective_features,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Cohort Size",
                "Cohort Size Perc",
                "Nonlinearity Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "Pred features ACC",
                "Prog features ACC",
                "Select features ACC",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / f"results/cohort_size_sensitivity/{self.synthetic_simulator_type}"
        )

        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}-seed{self.seed}.csv"
        )

        if self.visualize_progression:
            # Create GIF
            imageio.mimsave("progression.gif", frames, fps = 1)
            imageio.mimsave(results_path / "progression.gif", frames, fps=1)  # Set fps=1 for slower transition to observe changes clearly
            
