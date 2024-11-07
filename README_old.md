# Experiment Setup and Execution

This document provides the necessary steps to reproduce the experiments and plots from the paper.

## 1. Create Python Virtual Environment and Install Dependencies

To ensure the consistency of the experimental environment, please use Python 3.10.12. Follow the steps below:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Update Configuration Files

Before running the experiments, update the path information in the configuration files `config_TY_tcga.yaml` and `config_TY_toy.yaml`. Set the `repo_path` to the location of the submission code.

```yaml
repo_path = PATH_TO_SUBMISSION_CODE
```

## 3. Run Experiments

To execute the experiments, use the following commands from within the submission code folder:

```bash
python 01_run_simulation_experiment.py -cn=config_AY_tcga
python 01_run_simulation_experiment.py -cn=config_AY_toy
```

If you want to run the experiments with a filtered selection of models or seeds, modify the corresponding parameters ```model_names``` and ```seeds``` in config_TY_tcga.yaml and config_TY_toy.yaml. 

If you want to run the tcga experiment without evaluation of feature explainability, set ```evaluate_explanations: false``` and ```evaluate_prog_explanations: false```.

## 4. Plot Results

After the experiments are complete, you can generate the plots using:

```bash
python 02_plot_results.py
```