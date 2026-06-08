# IBM HR Attrition MLOps Project

This project implements an end-to-end MLOps pipeline for predicting employee attrition using the IBM HR Analytics Employee Attrition & Performance dataset.

## Project Structure

- `src/` - source code for preprocessing, training, evaluation, and monitoring
- `configs/` - YAML configuration files
- `tests/` - pytest test suite
- `.github/workflows/` - GitHub Actions CI/CD pipeline
- `data/` - raw and processed datasets
- `reports/` - drift and evaluation reports

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Training

Training script will be added in `src/train.py`.

## Run Tests

```bash
pytest tests/ -v
```

## Dataset

The dataset used is the IBM HR Analytics Employee Attrition & Performance dataset. The target variable is `Attrition`.

