from src.data_preprocessing import load_raw_data


def test_dataset_contains_expected_columns():
    df = load_raw_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    expected_columns = {
        "Age",
        "Attrition",
        "BusinessTravel",
        "DailyRate",
        "Department",
        "DistanceFromHome",
        "Education",
        "EmployeeCount",
        "Gender",
        "JobRole",
        "MonthlyIncome",
        "OverTime",
        "TotalWorkingYears",
    }

    assert expected_columns.issubset(set(df.columns))


def test_target_column_contains_only_expected_values():
    df = load_raw_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    allowed_values = {"Yes", "No"}
    actual_values = set(df["Attrition"].dropna().unique())

    assert actual_values.issubset(allowed_values)


def test_numeric_columns_are_within_reasonable_ranges():
    df = load_raw_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    assert df["Age"].between(18, 100).all()
    assert df["MonthlyIncome"].between(0, 100000).all()
    assert df["DistanceFromHome"].between(0, 1000).all()

    