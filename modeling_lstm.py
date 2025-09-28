import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_FILE = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\processed\xplane_features.csv"
MODEL_OUT = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\xplane_lstm.h5"

def create_sequences(X, y, timesteps=50):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No features file found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    #drop non-numeric columns
    df = df.select_dtypes(include=[np.number])

    #features and target
    X = df.drop(columns=["failure"]).values
    y = df["failure"].values

    #scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #create sequences
    TIMESTEPS = 50
    X_seq, y_seq = create_sequences(X_scaled, y, TIMESTEPS)

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    #LSTM Model
    model = Sequential([
        LSTM(64, input_shape=(TIMESTEPS, X_train.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    #train with early stopping
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=20, batch_size=64, callbacks=[es])

    #predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    #eval
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 LSTM Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    #save model
    model.save(MODEL_OUT)
    print(f"\n✅ LSTM model saved at: {MODEL_OUT}")

if __name__ == "__main__":
    main()
