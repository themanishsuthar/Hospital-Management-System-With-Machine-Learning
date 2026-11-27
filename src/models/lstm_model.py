"""
Long Short-Term Memory (LSTM) Neural Network Forecasting Model (FR6, FR7).
Aligned with SRS Chapter 5.4 Class Diagram.

Provides deep-learning sequence-to-one and multi-step recursive forecasting for
hospital patient load time-series, with feature scaling, windowing, and early stopping.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Suppress excessive TensorFlow output
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from src.models.base import ForecastModel, calculate_metrics


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class LSTMModel(ForecastModel):
    """
    Recurrent Neural Network model using LSTM cells for time-series forecasting.
    Transforms multivariate time-series data into 3D sliding window tensors:
    [samples, window_size, num_features].
    """

    def __init__(
        self,
        name: str = "LSTM",
        target_col: str = "opd_patients",
        feature_cols: Optional[List[str]] = None,
        window_size: int = 14,
        lstm_units: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 7,
        seed: int = 42,
    ):
        super().__init__(name=name, target_col=target_col, feature_cols=feature_cols)
        self.window_size = int(window_size)
        self.lstm_units = tuple(lstm_units)
        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.seed = int(seed)

        self.model: Optional[Sequential] = None
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.last_window_features_: Optional[np.ndarray] = None
        self.last_date_: Optional[pd.Timestamp] = None
        self.training_history_: Optional[Dict[str, List[float]]] = None

    def _build_model(self, num_features: int) -> Sequential:
        """Constructs and compiles the Keras Sequential LSTM architecture."""
        set_seed(self.seed)
        model = Sequential()
        model.add(Input(shape=(self.window_size, num_features)))

        num_layers = len(self.lstm_units)
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < num_layers - 1)
            model.add(LSTM(units=units, return_sequences=return_sequences))
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=1))  # Predict single continuous patient count

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
        return model

    def _create_sequences(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sliding window sequence pairs:
        X: shape (samples - window_size, window_size, features)
        y: shape (samples - window_size,)
        """
        x_seq, y_seq = [], []
        for i in range(self.window_size, len(x_data)):
            x_seq.append(x_data[i - self.window_size : i, :])
            y_seq.append(y_data[i])
        return np.array(x_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> "LSTMModel":
        """
        Fits MinMaxScaler on training features & target, creates sliding sequences,
        and trains the LSTM network with EarlyStopping.
        """
        set_seed(self.seed)
        df = train_df.sort_values("date").reset_index(drop=True)

        # Resolve features
        if not self.feature_cols:
            self.feature_cols = [
                c for c in df.columns
                if c not in ["date", self.target_col, "hospital_id", "ownership_type"]
                and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not self.feature_cols:
            raise ValueError("No numeric feature columns found or specified for LSTMModel.")

        if len(df) <= self.window_size:
            raise ValueError(
                f"Training dataset length ({len(df)}) must be greater than window_size ({self.window_size})."
            )

        x_raw = df[self.feature_cols].values.astype(np.float32)
        y_raw = df[[self.target_col]].values.astype(np.float32)

        # Fit scalers on training partition only to prevent data leakage
        x_scaled = self.scaler_x.fit_transform(x_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw)

        x_train_seq, y_train_seq = self._create_sequences(x_scaled, y_scaled)

        # Validation sequence generation
        validation_data = None
        callbacks = []
        if self.patience > 0:
            monitor_metric = "val_loss" if val_df is not None else "loss"
            callbacks.append(
                EarlyStopping(
                    monitor=monitor_metric,
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )

        if val_df is not None and not val_df.empty:
            v_df = val_df.sort_values("date").reset_index(drop=True)
            # Prepend the last (window_size) points from train_df to cover the initial validation boundary
            combined_v_df = pd.concat([df.tail(self.window_size), v_df], ignore_index=True)
            vx_raw = combined_v_df[self.feature_cols].values.astype(np.float32)
            vy_raw = combined_v_df[[self.target_col]].values.astype(np.float32)

            vx_scaled = self.scaler_x.transform(vx_raw)
            vy_scaled = self.scaler_y.transform(vy_raw)
            x_val_seq, y_val_seq = self._create_sequences(vx_scaled, vy_scaled)
            validation_data = (x_val_seq, y_val_seq)

        num_features = len(self.feature_cols)
        self.model = self._build_model(num_features=num_features)

        history = self.model.fit(
            x_train_seq,
            y_train_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False,  # Preserve sequential order in time series
        )

        self.training_history_ = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        # Store the last window features and timestamp for recursive horizon forecasting
        full_df = pd.concat([df, val_df], ignore_index=True) if val_df is not None else df
        full_df = full_df.sort_values("date").reset_index(drop=True)
        self.last_window_features_ = self.scaler_x.transform(
            full_df[self.feature_cols].tail(self.window_size).values.astype(np.float32)
        )
        self.last_date_ = pd.to_datetime(full_df["date"].iloc[-1])
        self.is_fitted = True
        return self

    def _check_fitted(self) -> None:
        """Verifies that model has been trained before generating inferences."""
        if not self.is_fitted or self.model is None:
            raise ValueError(f"Model '{self.name}' is not fitted yet. Call fit() first.")

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for each row of test_df using a rolling feature window.
        If test_df provides sequences or rows, each row's window is constructed.
        """
        self._check_fitted()
        if len(test_df) == 0:
            return np.array([], dtype=float)

        x_raw = test_df[self.feature_cols].values.astype(np.float32)
        x_scaled = self.scaler_x.transform(x_raw)

        if len(test_df) <= self.window_size:
            # When evaluating row-by-row or small slices, pad with last_window_features_
            padded_scaled = np.vstack([self.last_window_features_, x_scaled])
            seqs = []
            for i in range(len(x_scaled)):
                seqs.append(padded_scaled[i : i + self.window_size, :])
            x_seq = np.array(seqs, dtype=np.float32)
        else:
            # Full sequence evaluation
            x_seq = []
            # Use last_window_features_ to allow predictions starting from index 0 of test_df
            padded_scaled = np.vstack([self.last_window_features_, x_scaled])
            for i in range(len(x_scaled)):
                x_seq.append(padded_scaled[i : i + self.window_size, :])
            x_seq = np.array(x_seq, dtype=np.float32)

        preds_scaled = self.model.predict(x_seq, verbose=0)
        preds_unscaled = self.scaler_y.inverse_transform(preds_scaled).flatten()
        return np.maximum(0.0, np.round(preds_unscaled, 1))

    def forecast_horizon(
        self,
        horizon_days: int = 7,
        future_features_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Multi-step recursive sequence forecasting across future horizon days.
        """
        self._check_fitted()
        if self.last_date_ is None or self.last_window_features_ is None:
            raise ValueError("Model has not captured recent temporal state. Call fit() first.")

        forecast_dates = pd.date_range(
            start=self.last_date_ + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )

        current_window = self.last_window_features_.copy()
        predictions = []

        for step in range(horizon_days):
            # Predict next step from current_window [1, window_size, features]
            input_seq = np.expand_dims(current_window, axis=0)
            next_pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]
            next_pred_unscaled = float(
                self.scaler_y.inverse_transform([[next_pred_scaled]])[0, 0]
            )
            predictions.append(max(0.0, round(next_pred_unscaled, 1)))

            # Prepare next window feature vector
            if future_features_df is not None and len(future_features_df) > step:
                next_raw = future_features_df.iloc[[step]][self.feature_cols].values.astype(np.float32)
                next_scaled = self.scaler_x.transform(next_raw)[0]
            else:
                # Use rolling average of features or recent feature vector
                next_scaled = current_window[-1].copy()

            # Advance sliding window by 1 step
            current_window = np.vstack([current_window[1:], next_scaled])

        pred_col = f"predicted_{self.target_col.replace('_patients', '')}" if "patients" in self.target_col else f"predicted_{self.target_col}"

        return pd.DataFrame({
            "date": forecast_dates,
            pred_col: predictions,
        })

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves model weights, architecture, scalers, and metadata."""
        self._check_fitted()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        keras_path = path.with_suffix(".keras")
        meta_path = path.with_suffix(".meta.joblib")

        self.model.save(keras_path)
        metadata = {
            "name": self.name,
            "target_col": self.target_col,
            "feature_cols": self.feature_cols,
            "window_size": self.window_size,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "seed": self.seed,
            "scaler_x": self.scaler_x,
            "scaler_y": self.scaler_y,
            "last_window_features_": self.last_window_features_,
            "last_date_": self.last_date_,
            "training_history_": self.training_history_,
        }
        joblib.dump(metadata, meta_path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LSTMModel":
        """Loads model weights, scalers, and configuration state."""
        path = Path(filepath)
        keras_path = path.with_suffix(".keras")
        meta_path = path.with_suffix(".meta.joblib")

        if not keras_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing model files at {keras_path} or {meta_path}")

        meta = joblib.load(meta_path)
        instance = cls(
            name=meta["name"],
            target_col=meta["target_col"],
            feature_cols=meta["feature_cols"],
            window_size=meta["window_size"],
            lstm_units=meta["lstm_units"],
            dropout_rate=meta["dropout_rate"],
            learning_rate=meta["learning_rate"],
            batch_size=meta["batch_size"],
            epochs=meta["epochs"],
            patience=meta["patience"],
            seed=meta["seed"],
        )
        instance.scaler_x = meta["scaler_x"]
        instance.scaler_y = meta["scaler_y"]
        instance.last_window_features_ = meta["last_window_features_"]
        instance.last_date_ = meta["last_date_"]
        instance.training_history_ = meta.get("training_history_")
        instance.model = load_model(keras_path)
        instance.is_fitted = True
        return instance
