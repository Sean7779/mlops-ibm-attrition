# IBM HR Attrition MLOps Project

This project implements an end-to-end MLOps pipeline for predicting employee attrition using the IBM HR Analytics Employee Attrition & Performance dataset. The target variable is `Attrition`.

## Project Structure

- `src/` – source code for preprocessing, training, evaluation, and monitoring
- `configs/` – YAML configuration files
- `tests/` – pytest test suite
- `.github/workflows/` – GitHub Actions CI pipeline
- `data/` – raw and processed datasets
- `reports/` – drift and evaluation reports

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Run tests:

```bash
pytest tests/ -v
```

Train model:

```bash
python src/train.py
```

CI runs `pytest tests/ -v` on every push to `main` (see `.github/workflows/ci.yml`).

Experiment tracking with MLflow
This project uses MLflow to track model training runs and compare different hyperparameters.

src/train.py trains a logistic regression model on the IBM attrition dataset. It logs:

The trained sklearn model as an MLflow artifact.

Classification metrics such as accuracy, F1 score and ROC AUC.

Key parameters including max_iter, split_random_state, and model_random_state.

src/run_experiments.py runs the training pipeline multiple times with different configurations. For each configuration it:

Calls train_model with overridden max_iter and split_random_state.

Logs the chosen hyperparameters to MLflow as parameters.

Logs the resulting evaluation metrics to MLflow as metrics.

Tags the runs with experiment_group="sprint17" and model_type="logistic_regression".

src.compare_experiments.py programmatically compares the recorded runs. It:

Uses mlflow.get_experiment_by_name("ibm_attrition_experiments") to locate the experiment.

Calls mlflow.search_runs() to load all runs for that experiment.

Sorts the runs by accuracy in descending order.

Prints a small table of runs and highlights the best run.

Saves the comparison as reports/experiment_comparison.csv for offline inspection
To reproduce:

# 1. Single training run
python -m src.train

# 2. Run multiple experiments
python -m src.run_experiments

# 3. Compare experiments and write CSV report
python -m src.compare_experiments
You can also start the MLflow UI locally to inspect the runs and artifacts:

mlflow ui
and open http://127.0.0.1:5000 in a browser.








