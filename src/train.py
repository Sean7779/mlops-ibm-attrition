import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow
import mlflow.sklearn



from src.evaluate import evaluate_classification


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Loaded dataframe from {path} is empty.")

    return df


def prepare_features_and_target(df: pd.DataFrame, target_column: str = "Attrition"):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    y = y.map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Target column contains unexpected values.")

    return X, y


def train_model(
    data_path: str = None,
    config_path: str = "configs/config.yaml",
    log_to_mlflow: bool = False,
    max_iter_override: int = None,
    split_random_state_override: int = None,
):
    config = load_config(config_path)

    if data_path is None:
        data_path = config["data"]["raw_data_path"]

    target_column = config["data"]["target_column"]
    test_size = config["split"]["test_size"]
    split_random_state = config["split"]["random_state"]
    max_iter = config["model"]["max_iter"]
    model_random_state = config["model"]["random_state"]

    if max_iter_override is not None:
        max_iter = max_iter_override

    if split_random_state_override is not None:
        split_random_state = split_random_state_override

    df = load_training_data(data_path)
    X, y = prepare_features_and_target(df, target_column=target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=split_random_state,
        stratify=y,
    )

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=max_iter, random_state=model_random_state),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_classification(y_test, y_pred, y_proba)

    if log_to_mlflow:
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "data_path": data_path,
                    "target_column": target_column,
                    "test_size": test_size,
                    "split_random_state": split_random_state,
                    "max_iter": max_iter,
                    "model_random_state": model_random_state,
                }
            )

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.set_tag("model_type", "logistic_regression")
            mlflow.sklearn.log_model(model, artifact_path="model")

    return model, metrics, X_test, y_test


if __name__ == "__main__":
    train_model(log_to_mlflow=True)




    





