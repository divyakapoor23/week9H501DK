# ...existing code...
import pandas as pd
import numpy as np

class GroupEstimate:
    def __init__(self, estimate="mean"):
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates_ = None
        self.feature_names_ = None
        self.default_category_ = None
        self.default_estimates_ = None

    def fit(self, X, y, default_category=None):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = pd.Series(y, name="target")

        if len(X_df) != len(y_series):
            raise ValueError("X and y must have the same length")
        if y_series.isna().any():
            raise ValueError("y contains missing values")

        self.feature_names_ = list(X_df.columns)

        if default_category is not None:
            if default_category not in self.feature_names_:
                raise ValueError("default_category must be a column in X")
            self.default_category_ = default_category
        else:
            self.default_category_ = None

        combined = X_df.copy()
        combined["target"] = y_series.values

        agg = "mean" if self.estimate == "mean" else "median"
        if agg == "mean":
            self.group_estimates_ = combined.groupby(self.feature_names_, observed=True)["target"].mean()
        else:
            self.group_estimates_ = combined.groupby(self.feature_names_, observed=True)["target"].median()

        if self.default_category_ is not None:
            if agg == "mean":
                self.default_estimates_ = combined.groupby(self.default_category_, observed=True)["target"].mean()
            else:
                self.default_estimates_ = combined.groupby(self.default_category_, observed=True)["target"].median()
        else:
            self.default_estimates_ = None

        return self

    def predict(self, X_):
        if self.group_estimates_ is None:
            raise ValueError("Model must be fitted before calling predict")

        # Normalize input
        if isinstance(X_, pd.DataFrame):
            X_pred = X_.copy()
            # allow column order differences
            if set(X_pred.columns) != set(self.feature_names_):
                raise ValueError("Feature names in X_ do not match training data")
            X_pred = X_pred[self.feature_names_]
        else:
            X_pred = pd.DataFrame(X_, columns=self.feature_names_)

        # Build lookup maps
        if isinstance(self.group_estimates_.index, pd.MultiIndex):
            full_map = {tuple(idx): float(val) for idx, val in self.group_estimates_.items()}
        else:
            full_map = {(idx,): float(val) for idx, val in self.group_estimates_.items()}

        default_map = None
        if self.default_estimates_ is not None:
            default_map = {k: float(v) for k, v in self.default_estimates_.items()}

        preds = []
        missing = 0
        for _, row in X_pred.iterrows():
            key = tuple(row[col] for col in self.feature_names_)
            val = full_map.get(key)
            if val is None and default_map is not None:
                val = default_map.get(row[self.default_category_])
            if val is None:
                val = np.nan
                missing += 1
            preds.append(val)

        if missing:
            print(f"Warning: {missing} observations belong to groups not seen in training data")

        return np.asarray(preds, dtype=float)

    def get_group_estimates(self):
        return self.group_estimates_
# ...existing code...
