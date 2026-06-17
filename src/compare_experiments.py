import os
import mlflow


def main():
    os.makedirs("reports", exist_ok=True)

    experiment = mlflow.get_experiment_by_name("ibm_attrition_experiments")
    if experiment is None:
        raise ValueError("Experiment 'ibm_attrition_experiments' not found. Run experiments first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )

    if runs.empty:
        raise ValueError("No MLflow runs found.")

    columns = [
        "run_id",
        "metrics.accuracy",
        "metrics.f1_score",
        "metrics.roc_auc",
        "params.max_iter",
        "params.split_random_state",
    ]

    available_columns = [col for col in columns if col in runs.columns]
    summary = runs[available_columns].copy()

    print(summary.head(10))
    print("\nBest run:")
    print(summary.iloc[0])

    summary.to_csv("reports/experiment_comparison.csv", index=False)


if __name__ == "__main__":
    main()

    