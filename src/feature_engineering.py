def create_features(df):

    df = df.copy()

    df["parent_edu"] = (
        df["Medu"] +
        df["Fedu"]
    )

    df["total_alcohol"] = (
        df["Dalc"] +
        df["Walc"]
    )

    df["study_failure_ratio"] = (
        df["studytime"] /
        (df["failures"] + 1)
    )

    return df