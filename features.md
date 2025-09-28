import pandas as pd

def inject_failures(df: pd.DataFrame, random_frac: float = 0.02) -> pd.DataFrame:
    """
    Create synthetic failure labels.
    1. Try threshold-based rules.
    2. If still all 0s, inject random failures for training/demo purposes.
    """
    df["failure"] = 0  #start with all good

    #threshold-based failures
    if "N1__1,_pcnt" in df.columns and "N1__2,_pcnt" in df.columns:
        df.loc[(df["N1__1,_pcnt"] > 80) | (df["N1__2,_pcnt"] > 80), "failure"] = 1

    if "OILT1,__deg" in df.columns and "OILT2,__deg" in df.columns:
        df.loc[(df["OILT1,__deg"] > 150) | (df["OILT2,__deg"] > 150), "failure"] = 1

    if "EGT_1,__deg" in df.columns and "EGT_2,__deg" in df.columns:
        df.loc[(df["EGT_1,__deg"] > 700) | (df["EGT_2,__deg"] > 700), "failure"] = 1

    #fallback: inject random failures if all 0s
    if df["failure"].sum() == 0:
        failure_indices = df.sample(frac=random_frac, random_state=42).index
        df.loc[failure_indices, "failure"] = 1
        print(f"⚠️ No threshold failures found → injected {len(failure_indices)} random failures.")

    return df

def main():
    #load processed data
    df = pd.read_csv(r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\processed\xplane_cleaned.csv")

    #inject synthetic failures
    df = inject_failures(df)

    #save engineered features
    df.to_csv(r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\processed\xplane_features.csv", index=False)
    print(f"✅ Features with synthetic failures saved at: data/processed/xplane_features.csv")
    print(df["failure"].value_counts()) #to show class balance

if __name__ == "__main__":
    main()
