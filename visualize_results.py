# visualize_results.py

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from yassify_football_pipeline import (
    load_match_data,
    encode_labels,
    create_sequences,
    split_data,
    scale_data,
    recursive_forecast
)


def plot_rolling_features(df, features=None):
    """Plot rolling-window features over each match index."""
    if features is None:
        features = [c for c in df.columns if c.startswith("roll_")]
    plt.figure(figsize=(12, 8))
    for feat in features:
        plt.plot(df.index, df[feat], label=feat)
    plt.title("Rolling Features Over Matches")
    plt.xlabel("Match Index")
    plt.ylabel("Normalized Value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, labels):
    """Display the confusion matrix for classification results."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    plt.title("Holdout Confusion Matrix")
    plt.show()


def plot_goals_predictions(actual, predicted):
    """Compare true vs predicted goals on the holdout set."""
    plt.figure(figsize=(10, 6))
    plt.plot(actual, marker="o", label="Actual Goals")
    plt.plot(predicted, marker="x", label="Predicted Goals")
    plt.title("Holdout: Actual vs Predicted Goals")
    plt.xlabel("Holdout Match Index")
    plt.ylabel("Goals Scored")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained pipeline and show key charts"
    )
    parser.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        default=Path("barcelona_last_30_matches.json"),
        help="JSON file of match data (default: barcelona_last_30_matches.json)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=5,
        help="Rolling-window length (default: 5)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Validation fraction (default: 0.2)"
    )
    parser.add_argument(
        "--holdout", type=int, default=2,
        help="Number of holdout matches (default: 2)"
    )
    parser.add_argument(
        "--models_dir", type=Path, default=Path("./models"),
        help="Directory where clf.keras & reg.keras live"
    )
    args = parser.parse_args()

    # 1) Load & preprocess
    df = load_match_data(args.data_file, rolling_window=args.timesteps)
    encoder = encode_labels(df)
    X, y_res, y_goals = create_sequences(df, timesteps=args.timesteps)
    splits = split_data(X, y_res, y_goals, holdout=args.holdout, val_frac=args.test_size)
    Xtr_s, Xv_s, Xh_s, scaler = scale_data(
        splits['X_train'], splits['X_val'], splits['X_hold']
    )

    # 2) Load trained models
    import tensorflow as tf
    clf = tf.keras.models.load_model(args.models_dir / 'clf.keras')
    reg = tf.keras.models.load_model(args.models_dir / 'reg.keras')

    # 3) Get holdout predictions
    y_pred_res = np.argmax(clf.predict(Xh_s), axis=1)
    y_true_res = splits['y_res_hold']
    y_pred_goals = reg.predict(Xh_s).flatten()
    y_true_goals = splits['y_goals_hold']

    # 4) Plot everything
    plot_rolling_features(df)
    plot_confusion(y_true_res, y_pred_res, encoder.classes_)
    plot_goals_predictions(y_true_goals, y_pred_goals)

    # 5) Optional: recursive forecast printout
    print("\nRecursive forecast for next 2 matches:")
    recursive_forecast(clf, reg, Xh_s[0], scaler, encoder, steps=2)
