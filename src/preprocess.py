def remove_leakage(df):
    df = df.copy()
    df = df.drop(columns=["G1", "G2"])
    return df