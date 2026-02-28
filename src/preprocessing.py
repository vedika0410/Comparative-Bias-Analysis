import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def create_binary_columns(df):
    # Income binary
    df["income_binary"] = df["income"].apply(
        lambda x: 1 if ">50K" in str(x) else 0
    )

    # Gender binary (Male = 1)
    df["gender_binary"] = df["gender"].apply(
        lambda x: 1 if str(x).strip() == "Male" else 0
    )

    # Age binary (Older = 1 if age >= 40)
    df["age_old_binary"] = df["age"].apply(
        lambda x: 1 if x >= 40 else 0
    )

    return df


def occupation_level(occ):
    high_income_jobs = ["Exec-managerial", "Prof-specialty", "Tech-support"]
    return "high earning job" if occ in high_income_jobs else "regular job"


def create_text_features(df):
    df["hard_text"] = (
        df["age"].astype(str) + " year old " +
        df["gender"].astype(str) + " with " +
        df["education"].astype(str) +
        " working as a " + df["occupation"].astype(str) +
        " (" + df["occupation"].apply(occupation_level) + ")" +
        " having capital gain " + df["capital-gain"].astype(str) +
        " and working " + df["hours-per-week"].astype(str) + " hours weekly."
    )
    return df