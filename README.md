
# Learning Personalized Treatment Decisions in Precision Medicine

This repository accompanies the paper [*Learning Personalized Treatment Decisions in Precision Medicine: Disentangling Treatment Assignment Bias in Counterfactual Outcome Prediction and Biomarker Identification*](https://arxiv.org/abs/2410.00509), accepted at the ML4H 2024 conference. It provides code for replicating experiments, analyzing treatment assignment biases, and exploring counterfactual outcome prediction and biomarker identification. 

## Overview

Precision medicine offers the potential to tailor treatment strategies to individual patients based on their unique characteristics. However, leveraging clinical observational data for personalized treatment decisions poses significant challenges due to inherent biases and the high-dimensional nature of biological data.

This study aims to model various types of treatment assignment biases, using mutual information to quantify their impacts on machine learning models for counterfactual prediction and biomarker identification. Unlike standard counterfactual benchmarks that rely on fixed treatment policies, this work explores the characteristics of observational treatment policies across different clinical settings. This approach helps in understanding how different biases affect model performance in predicting outcomes and identifying biomarkers.

### Key Insights:
- **Treatment Assignment Policies**: These policies vary based on clinical settings, influencing how treatment decisions are made. For instance, regulated environments like cancer care often have systematic treatment assignments initially but may become more variable as patient care progresses.
- **Overlap Assumption Violation**: Observational data may violate the overlap assumption, where patients do not have an equal chance of receiving all treatment types. High-dimensional data exacerbates this issue, making it essential to identify when biases are impactful.
- **Empirical vs. Synthetic Data**: Using both synthetic (e.g., TCGA data) and real-world datasets (e.g., CRISPR and drug screens from DepMap), the study emphasizes a more biologically realistic approach by incorporating empirical biological mechanisms.

### Visual Illustration:
Figure 1 below (from the associated paper) visually represents different clinical settings and their respective treatment assignment policies, highlighting the potential outcomes and the violations of the overlap assumption that complicate counterfactual prediction. The figure helps illustrate how selection biases impact treatment decision-making and predictive model development.

![Figure 1](assets/bias_visualization.pdf)  <!-- Ensure the PDF is in the correct path -->


Precision medicine promises tailored treatment strategies but faces challenges due to biases in clinical observational data and the complexity of biological data. This work models various treatment assignment biases and investigates their effects on machine learning models for counterfactual outcome prediction and biomarker identification.

### Key Features
- Simulates different treatment policies and biases.
- Evaluates counterfactual prediction performance using synthetic, semi-synthetic (e.g., TCGA data), and real-world datasets (CRISPR and drug screens from DepMap).
- Quantifies the effect of treatment selection bias on model performance.
- Identifies and assesses biomarkers based on their predictive and prognostic capabilities.


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

Modify parameters like `model_names` and `seeds` in the configuration files if needed.

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
