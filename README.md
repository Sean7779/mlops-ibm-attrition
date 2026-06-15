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








