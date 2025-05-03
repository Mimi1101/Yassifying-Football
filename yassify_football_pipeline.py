import argparse
import json
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, mean_squared_error
from datetime import datetime

import joblib
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

import matplotlib.pyplot as plt


def set_random_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_match_data(file_path: Path, rolling_window: int = 5) -> pd.DataFrame:
    raw = pd.json_normalize(json.loads(file_path.read_text()))
    df = pd.DataFrame({
        'home': raw['home_or_away'].map({'Home': 1, 'Away': 0}),
        'expected_goals': pd.to_numeric(raw['stats.expected_goals'], errors='coerce'),
        'shots_on_goal': raw['stats.Shots on Goal'].fillna(0).astype(int),
        'total_shots': raw['stats.Total Shots'].fillna(0).astype(int),
        'possession': raw['stats.Ball Possession'].str.rstrip('%').astype(float).fillna(0),
        'pass_accuracy': raw['stats.Passes %'].str.rstrip('%').astype(float).fillna(0),
        'corner_kicks': raw['stats.Corner Kicks'].fillna(0).astype(int),
        'barca_goals': raw['barca_goals'].astype(int),
        'result': raw['result'],
    })

    df['result_score'] = df['result'].map({'Win':1.0, 'Draw':0.5, 'Loss':0.0})

    for col in [
        'expected_goals','shots_on_goal','total_shots','possession',
        'pass_accuracy','corner_kicks','result_score','barca_goals'
    ]:
        df[f'roll_{col}'] = (
            df[col]
              .rolling(window=rolling_window, min_periods=1)
              .mean()
              .shift(1)
        )

    df = df.dropna(subset=[f'roll_{c}' for c in [
        'expected_goals','shots_on_goal','total_shots',
        'possession','pass_accuracy','corner_kicks',
        'result_score','barca_goals'
    ]]).reset_index(drop=True)

    return df


def encode_labels(df: pd.DataFrame) -> LabelEncoder:
    enc = LabelEncoder()
    df['result_encoded'] = enc.fit_transform(df['result'])
    logging.info("Encoded result classes: %s", enc.classes_)
    return enc


def create_sequences(
    df: pd.DataFrame,
    timesteps: int = 5,
    feature_cols: list[str] = None
):
    if feature_cols is None:
        feature_cols = [
            'home',
            'roll_expected_goals','roll_shots_on_goal','roll_total_shots',
            'roll_possession','roll_pass_accuracy','roll_corner_kicks',
            'roll_result_score','roll_barca_goals'
        ]
    Xr = df[feature_cols].to_numpy()
    yr = df['result_encoded'].to_numpy()
    yg = df['barca_goals'].to_numpy()

    X, y_res, y_goals = [], [], []
    for i in range(len(Xr) - timesteps):
        X.append(Xr[i:i+timesteps])
        y_res.append(yr[i+timesteps])
        y_goals.append(yg[i+timesteps])
    return np.array(X), np.array(y_res), np.array(y_goals)


def split_data(
    X, y_res, y_goals,
    holdout: int = 2,
    val_frac: float = 0.2
):
    X_all = X[:-holdout]
    y_res_all = y_res[:-holdout]
    y_goals_all = y_goals[:-holdout]

    n_val = int(len(X_all) * val_frac)
    split = len(X_all) - n_val

    X_train = X_all[:split]
    X_val   = X_all[split:]
    y_res_train = y_res_all[:split]
    y_res_val   = y_res_all[split:]
    y_goals_train = y_goals_all[:split]
    y_goals_val   = y_goals_all[split:]

    X_hold = X[-holdout:]
    y_res_hold = y_res[-holdout:]
    y_goals_hold = y_goals[-holdout:]

    logging.info("Chrono split: train=%s, val=%s, holdout=%s",
                 X_train.shape, X_val.shape, X_hold.shape)
    return dict(
        X_train=X_train, X_val=X_val, X_hold=X_hold,
        y_res_train=y_res_train, y_res_val=y_res_val, y_res_hold=y_res_hold,
        y_goals_train=y_goals_train, y_goals_val=y_goals_val, y_goals_hold=y_goals_hold
    )


def scale_data(X_train, X_val, X_hold):
    _, T, F = X_train.shape
    scaler = MinMaxScaler()
    scaler.fit(X_train.reshape(-1, F))

    def _s(X):
        return scaler.transform(X.reshape(-1, F)).reshape(X.shape)

    return _s(X_train), _s(X_val), _s(X_hold), scaler


def build_models(timesteps: int, n_features: int, n_classes: int):
    clf = Sequential([
        tf.keras.Input((timesteps, n_features)),
        LSTM(32, return_sequences=True, dropout=0.5),
        LSTM(16, dropout=0.5),
        Dense(16, activation='relu'),
        Dense(n_classes, activation='softmax')
    ], name='clf')
    clf.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

    reg = Sequential([
        tf.keras.Input((timesteps, n_features)),
        LSTM(32, return_sequences=True, dropout=0.5),
        LSTM(16, dropout=0.5),
        Dense(16, activation='relu'),
        Dense(1)
    ], name='reg')
    reg.compile('adam', 'mse', metrics=['mae'])

    return clf, reg


def recursive_forecast(
    model_clf, model_reg,
    last_window, scaler, encoder,
    feature_names: list[str],
    steps: int = 2
):
    seq = last_window.copy()
    F = seq.shape[-1]
    for _ in range(steps):
        inp = seq[np.newaxis]
        p = model_clf.predict(inp)[0]
        g = model_reg.predict(inp)[0,0]
        idx = np.argmax(p)
        lbl = encoder.inverse_transform([idx])[0]
        print(f"→ Predicted: {lbl!r} ({p[idx]:.2f}) & goals={g:.1f}")

        raw_window = scaler.inverse_transform(seq.reshape(-1, F)).reshape(seq.shape)
        last_raw = raw_window[-1].copy()
        last_raw[ feature_names.index('roll_result_score') ] = {'Win':1,'Draw':0.5,'Loss':0}[lbl]
        last_raw[ feature_names.index('roll_barca_goals') ]    = g

        raw_window = np.vstack([raw_window[1:], last_raw])
        seq = scaler.transform(raw_window.reshape(-1, F)).reshape(seq.shape)


def get_recursive_forecast(
    model_clf, model_reg,
    last_window, scaler, encoder,
    feature_names: list[str],
    steps: int = 2
):
    """
    Returns (labels, numeric_scores, predicted_goals) for the next `steps` matches.
    """
    seq = last_window.copy()
    F = seq.shape[-1]
    labels, scores, goals = [], [], []

    for _ in range(steps):
        inp = seq[np.newaxis]
        p = model_clf.predict(inp)[0]
        g = model_reg.predict(inp)[0,0]
        idx = p.argmax()
        lbl = encoder.inverse_transform([idx])[0]
        score = {'Win':1.0,'Draw':0.5,'Loss':0.0}[lbl]

        labels.append(lbl)
        scores.append(score)
        goals.append(g)

        raw = scaler.inverse_transform(seq.reshape(-1, F)).reshape(seq.shape)
        last = raw[-1].copy()
        last[feature_names.index('roll_result_score')] = score
        last[feature_names.index('roll_barca_goals')]   = g
        raw = np.vstack([raw[1:], last])
        seq = scaler.transform(raw.reshape(-1, F)).reshape(seq.shape)

    return labels, scores, goals


def train_and_evaluate(
    model_clf, model_reg, splits,
    encoder, scaler, output_dir: Path,
    epochs: int, batch_size: int
):
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'hyperparams.json','w') as f:
        json.dump({'epochs':epochs,'batch_size':batch_size}, f)

    ck1 = ModelCheckpoint(output_dir/'clf.keras',
                          save_best_only=True, monitor='val_accuracy')
    ck2 = ModelCheckpoint(output_dir/'reg.keras',
                          save_best_only=True, monitor='val_mae')
    es = EarlyStopping(patience=5, restore_best_weights=True)
    rp = ReduceLROnPlateau(patience=3, factor=0.5)
    tb = TensorBoard(log_dir=str(output_dir/'logs'))

    Xtr, Xv, Xh = splits['X_train_s'], splits['X_val_s'], splits['X_hold_s']
    yrt, yrv = splits['y_res_train'], splits['y_res_val']
    ygt, ygv = splits['y_goals_train'], splits['y_goals_val']

    model_clf.fit(
        Xtr, yrt,
        validation_data=(Xv, yrv),
        epochs=epochs, batch_size=batch_size,
        callbacks=[ck1, es, rp, tb]
    )
    model_reg.fit(
        Xtr, ygt,
        validation_data=(Xv, ygv),
        epochs=epochs, batch_size=batch_size,
        callbacks=[ck2, es, rp, tb]
    )

    joblib.dump(scaler, output_dir/'scaler.joblib')
    joblib.dump(encoder, output_dir/'encoder.joblib')

    # Hold-out evaluation
    p_probs = model_clf.predict(splits['X_hold_s'])
    p_idx   = np.argmax(p_probs, axis=1)
    p_lbl   = encoder.inverse_transform(p_idx)
    p_goals = model_reg.predict(splits['X_hold_s']).flatten()

    print("\n=== Holdout Classification Report ===")
    print(classification_report(
        splits['y_res_hold'], p_idx,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        zero_division=0
    ))
    rmse = np.sqrt(mean_squared_error(splits['y_goals_hold'], p_goals))
    print(f"\nHoldout Regression RMSE: {rmse:.3f}\n")

    print("** Recursive forecast for next 2 matches **")
    recursive_forecast(
        model_clf, model_reg,
        splits['X_hold_s'][0], scaler, encoder,
        feature_names, steps=2
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('file', type=Path,
                   default=Path('barcelona_last_30_matches.json'),
                   nargs='?')
    p.add_argument('--timesteps', type=int, default=5)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--holdout', type=int, default=2)
    p.add_argument('--output_dir', type=Path, default=Path('./models'))
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    set_random_seeds(42)

    df = load_match_data(args.file, rolling_window=args.timesteps)
    encoder = encode_labels(df)

    X, y_res, y_goals = create_sequences(df, timesteps=args.timesteps)
    splits = split_data(
        X, y_res, y_goals,
        holdout=args.holdout,
        val_frac=args.test_size
    )
    Xtr_s, Xv_s, Xh_s, scaler = scale_data(
        splits['X_train'], splits['X_val'], splits['X_hold']
    )
    splits.update(
        X_train_s=Xtr_s,
        X_val_s=Xv_s,
        X_hold_s=Xh_s
    )

    global feature_names
    feature_names = [
        'home',
        'roll_expected_goals','roll_shots_on_goal','roll_total_shots',
        'roll_possession','roll_pass_accuracy','roll_corner_kicks',
        'roll_result_score','roll_barca_goals'
    ]

    T, F = Xtr_s.shape[1], Xtr_s.shape[2]
    clf, reg = build_models(T, F, len(encoder.classes_))

    # Train, evaluate, and print hold-out + recursive forecast
    train_and_evaluate(
        clf, reg, splits,
        encoder, scaler,
        args.output_dir,
        args.epochs, args.batch_size
    )

    # --- Now get the same 2-step forecast and plot it ---
    last_window = splits['X_hold_s'][0]
    labels, scores, goals = get_recursive_forecast(
        clf, reg, last_window,
        scaler, encoder, feature_names,
        steps=2
    )

    matches = [f"Match {i+1}" for i in range(len(scores))]
    plt.figure()
    plt.bar(matches, scores)
    plt.ylim(0, 1.1)
    plt.ylabel("Result Score (1=Win, 0.5=Draw, 0=Loss)")
    plt.title("Model’s Predicted Result Scores for Next 2 Matches")
    for i, lbl in enumerate(labels):
        plt.text(i, scores[i] + 0.02, lbl, ha='center')
    plt.show()


if __name__ == '__main__':
    main()
