import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw CSV data from the given path.
    """
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Loaded dataframe from {path} is empty.")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataframe with missing values handled.

    Numeric columns are filled with the median.
    Categorical columns are filled with the mode.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned_df = df.copy(deep=True)

    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns
    categorical_columns = cleaned_df.select_dtypes(exclude=["number"]).columns

    for col in numeric_columns:
        median_value = cleaned_df[col].median()
        cleaned_df[col] = cleaned_df[col].fillna(median_value)

    for col in categorical_columns:
        mode_series = cleaned_df[col].mode()
        if not mode_series.empty:
            cleaned_df[col] = cleaned_df[col].fillna(mode_series[0])

    return cleaned_df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataframe with categorical columns one-hot encoded.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    encoded_df = df.copy(deep=True)

    categorical_columns = encoded_df.select_dtypes(exclude=["number"]).columns

    if len(categorical_columns) == 0:
        return encoded_df

    encoded_df = pd.get_dummies(
        encoded_df,
        columns=categorical_columns,
        drop_first=True
    )

    return encoded_df 








