
# Learning Personalized Treatment Decisions in Precision Medicine

This repository accompanies the paper [*Learning Personalized Treatment Decisions in Precision Medicine: Disentangling Treatment Assignment Bias in Counterfactual Outcome Prediction and Biomarker Identification*](https://arxiv.org/abs/2410.00509), accepted at the ML4H 2024 conference. It provides code for replicating experiments, analyzing treatment assignment biases, and exploring counterfactual outcome prediction and biomarker identification. 

## Overview

Precision medicine offers the potential to tailor treatment strategies to individual patients based on their unique characteristics. However, leveraging clinical observational data for personalized treatment decisions poses significant challenges due to inherent biases and the high-dimensional nature of biological data.

This study aims to model various types of treatment assignment biases, using mutual information to quantify their impacts on machine learning models for counterfactual prediction and biomarker identification. Unlike standard counterfactual benchmarks that rely on fixed treatment policies, this work explores the characteristics of observational treatment policies across different clinical settings. This approach helps in understanding how different biases affect model performance in predicting outcomes and identifying biomarkers.

### Contributions
- **Formalization and Quantification**: We formalize and quantify different types of treatment assignment biases induced by observational treatment policies and explain their relationships with clinical settings and biomarker types.
- **Simulation and Analysis**: We systematically simulate different types of treatment selection policies and analyze their impact on the performance of various state-of-the-art counterfactual ML models using toy, semi-synthetic, and real-world outcomes.
- **Novel Evaluation Approach**: We propose using in-vitro experiments for counterfactual evaluation, providing the community with a realistic evaluation approach characterized by empirical outcomes and multi-modal biological covariates.
- **Insights on Bias and Model Performance**: Our findings show that the type of bias significantly influences model performance. Importantly, the violation of the overlap assumption does not always harm prediction accuracy, highlighting the need for nuanced approaches in high-dimensional medical data.
- **Model Differentiation**: The results indicate that models respond differently to various biases, providing critical insights for developing new methodologies and algorithms tailored to specific clinical settings.

## Setup and Installation
If you have any questions about the implementation or content of the paper, feel free to reach out to me (michavol@ethz.ch).
Ensure you are using Python 3.10.12. Follow these steps to set up the environment:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Update the `config_TY_tcga.yaml` and `config_TY_toy.yaml` files with the appropriate path to the repository code:

```yaml
repo_path = PATH_TO_REPO
```

## Running the Experiments

Navigate to the submission code folder and run:

```bash
python 01_run_simulation_experiment.py -cn=config_AY_tcga
python 01_run_simulation_experiment.py -cn=config_AY_toy
```

Modify parameters like `model_names` and `seeds` in the configuration files if needed. The results can be found in the `results` directory specified in `config_TY_tcga.yaml` or `config_TY_toy.yaml`.

### Optional Settings
Disable feature explainability evaluation by setting:
  ```yaml
  evaluate_explanations: false
  evaluate_prog_explanations: false
  ```

## Generating Plots

After running the experiments, generate the results plots with:

```bash
python 02_plot_results.py
```

## Citing This Work

Please reference this repository and the associated paper when using or extending this code:
```
@misc{vollenweider2024learningpersonalizedtreatmentdecisions,
      title={Learning Personalized Treatment Decisions in Precision Medicine: Disentangling Treatment Assignment Bias in Counterfactual Outcome Prediction and Biomarker Identification}, 
      author={Michael Vollenweider and Manuel Schürch and Chiara Rohrer and Gabriele Gut and Michael Krauthammer and Andreas Wicki},
      year={2024},
      eprint={2410.00509},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.00509}, 
}
```
## Acknowledgments

Parts of the code in this repository are adapted from Alicia Curth's implementations of neural treatment effect models and the `EconML` library for the linear lasso models. Refer to their repositories for additional details:
- [Alicia Curth's code](https://github.com/AliciaCurth/CATENets)
- [EconML library](https://github.com/py-why/EconML)
