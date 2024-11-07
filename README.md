
# Learning Personalized Treatment Decisions in Precision Medicine

This repository accompanies the paper *Learning Personalized Treatment Decisions in Precision Medicine: Disentangling Treatment Assignment Bias in Counterfactual Outcome Prediction and Biomarker Identification*, accepted at the ML4H 2024 conference. It provides code for replicating experiments, analyzing treatment assignment biases, and exploring counterfactual outcome prediction and biomarker identification.

## Overview

Precision medicine promises tailored treatment strategies but faces challenges due to biases in clinical observational data and the complexity of biological data. This work models various treatment assignment biases and investigates their effects on machine learning models for counterfactual outcome prediction and biomarker identification.

### Key Features
- Simulates different treatment policies and biases.
- Evaluates counterfactual prediction performance using synthetic, semi-synthetic (e.g., TCGA data), and real-world datasets (CRISPR and drug screens from DepMap).
- Quantifies the effect of treatment selection bias on model performance.
- Identifies and assesses biomarkers based on their predictive and prognostic capabilities.


## Setup and Installation

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

Modify parameters like `model_names` and `seeds` in the configuration files if needed.

### Optional Settings
- Disable feature explainability evaluation by setting:
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
