import mlflow

from src.train import train_model


def main():
    mlflow.set_experiment("ibm_attrition_experiments")

    configs = [
        {"max_iter": 500, "split_random_state": 42},
        {"max_iter": 800, "split_random_state": 7},
        {"max_iter": 1000, "split_random_state": 21},
        {"max_iter": 1500, "split_random_state": 42},
        {"max_iter": 2000, "split_random_state": 99},
    ]

    for cfg in configs:
        with mlflow.start_run():
            _, metrics, _, _ = train_model(
                log_to_mlflow=False,
                max_iter_override=cfg["max_iter"],
                split_random_state_override=cfg["split_random_state"],
            )

            mlflow.log_param("max_iter", cfg["max_iter"])
            mlflow.log_param("split_random_state", cfg["split_random_state"])

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.set_tag("experiment_group", "sprint17")
            mlflow.set_tag("model_type", "logistic_regression")


if __name__ == "__main__":
    main()


    