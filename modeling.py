import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import joblib


def clean_data(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Clean dataset before modeling:
    - Drop unnamed/index-like columns
    - Keep only numeric + target
    """
    #drop unwanted unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    #keep numeric and target only
    keep_cols = df.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    if target not in keep_cols:
        keep_cols.append(target)

    return df[keep_cols]


def train_xgboost(df: pd.DataFrame, target: str = "failure"):
    """
    Train XGBoost classifier with imbalance handling.
    """
    #cleaning the data
    df = clean_data(df, target)

    #features and target
    X = df.drop(columns=[target])
    y = df[target]

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #handle imbalance
    neg, pos = np.bincount(y_train)
    scale = neg / pos
    print(f"⚖️ Class balance -> Negative: {neg}, Positive: {pos}, scale_pos_weight={scale:.2f}")

    #XGBoost model
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale,
        use_label_encoder=False
    )

    #train
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    best_acc, best_thresh = 0, 0.5
    for t in [i/100 for i in range(20, 90, 5)]:  # test thresholds 0.2–0.85
        acc = accuracy_score(y_test, (y_proba > t).astype(int))
        if acc > best_acc:
            best_acc, best_thresh = acc, t
        #predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    print(f"🎯 Best threshold={best_thresh}, Accuracy={best_acc:.4f}")

    #evaluation
    print("\n📊 Model Evaluation:")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

    print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
    print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, X_test, y_test, y_pred


def main():
    df = pd.read_csv(r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\processed\xplane_features.csv")

    #training the model
    model, X_test, y_test, y_pred = train_xgboost(df, target="failure")

    #saving the model
    joblib.dump(model, r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\xplane_xgboost.pkl")
    print("✅ Model saved at: models/xplane_xgboost.pkl")

if __name__ == "__main__":
    main()
