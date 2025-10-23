"""Utilities: GroupEstimate model for estimating target values based on feature groups.
This module defines the GroupEstimate class which can fit a model to compute
mean or median target values for groups defined by categorical features.
It also supports handling unseen groups during prediction by using default
category estimates.
"""


import pandas as pd
import numpy as np

class GroupEstimate:
    def __init__(self, estimate="mean"):
        """Initialize the GroupEstimate model."""
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates_ = None
        self.feature_names_ = None
        self.default_category_ = None
        self.default_estimates_ = None

    def fit(self, X, y, default_category=None):
        """Fit the GroupEstimate model."""
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = pd.Series(y, name="target")
        # Input validation
        if len(X_df) != len(y_series):
            raise ValueError("X and y must have the same length")
        if y_series.isna().any():
            raise ValueError("y contains missing values")

        self.feature_names_ = list(X_df.columns)
        # Handle default category
        if default_category is not None:
            if default_category not in self.feature_names_:
                raise ValueError("default_category must be a column in X")
            self.default_category_ = default_category
        else:
            self.default_category_ = None
        # Combine features and target
        combined = X_df.copy()
        combined["target"] = y_series.values
        # Compute group estimates
        agg = "mean" if self.estimate == "mean" else "median"
        # Group by features and compute the estimate
        if agg == "mean":
            self.group_estimates_ = combined.groupby(self.feature_names_, observed=True)["target"].mean()
        else:
            self.group_estimates_ = combined.groupby(self.feature_names_, observed=True)["target"].median()
        # Compute default estimates if applicable
        if self.default_category_ is not None:
            if agg == "mean":
                self.default_estimates_ = combined.groupby(self.default_category_, observed=True)["target"].mean()
            else:
                self.default_estimates_ = combined.groupby(self.default_category_, observed=True)["target"].median()
        else:
            self.default_estimates_ = None

        return self

    def predict(self, X_):
        """Predict using the GroupEstimate model."""
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
        # Handle default estimates
        if self.default_estimates_ is not None:
            default_map = {k: float(v) for k, v in self.default_estimates_.items()}

        # Generate predictions
        preds = []
        missing = 0
        # Iterate over each row in the prediction DataFrame
        for _, row in X_pred.iterrows():
            key = tuple(row[col] for col in self.feature_names_)
            val = full_map.get(key)
            # Check default category if needed
            if val is None and default_map is not None:
                val = default_map.get(row[self.default_category_])
            if val is None:
                val = np.nan
                missing += 1
            preds.append(val)
        # Print warning if there are missing predictions
        if missing:
            print(f"Warning: {missing} observations belong to groups not seen in training data")

        return np.asarray(preds, dtype=float)

    def get_group_estimates(self):
        """Get the group estimates after fitting."""
        return self.group_estimates_


if __name__ == "__main__":
    # Example from the notebook — prints a numpy array and its type
    X = pd.DataFrame({
        "loc_country": ["Guatemala", "Mexico"],
        "roast": ["Light", "Medium"]
    })
    y = [88.4, 91.0]

    gm = GroupEstimate(estimate="mean").fit(X, y)

    X_ = [
        ["Guatemala", "Light"],
        ["Mexico", "Medium"],
        ["Canada", "Dark"]
    ]

    pred = gm.predict(X_)
    print(pred)           # numpy array output
    print(type(pred))     # confirm it's numpy.ndarray
    # optional: show which entries are NaN
    print(np.isnan(pred).tolist())