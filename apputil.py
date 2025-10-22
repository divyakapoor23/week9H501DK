# ...existing code...
import pandas as pd
import numpy as np

class GroupEstimate:
    def __init__(self, estimate="mean"):
        """
        Initialize GroupEstimate with either "mean" or "median" as the estimate type.
        """
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates_ = None         
        self.feature_names_ = None
        self.default_category_ = None       
        self.default_estimates_ = None       
    def fit(self, X, y, default_category=None):
        """
        Fit the GroupEstimate model by calculating group estimates from training data.

        Parameters:
        X (pd.DataFrame or array-like): DataFrame of categorical features
        y (array-like): 1-D array of target values (no missing values)
        default_category (str, optional): column name in X to use as fallback when full
                                          category-combination is unseen at predict time.
        """
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_series = pd.Series(y, name="target")

        if len(X_df) != len(y_series):
            raise ValueError("X and y must have the same length")
        if y_series.isna().any():
            raise ValueError("y contains missing values")

        self.feature_names_ = list(X_df.columns)
        if default_category is not None:
            if default_category not in self.feature_names_:
                raise ValueError("default_category must be one of the columns in X")
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

        # If default_category provided, compute estimates grouped by that single column
        if self.default_category_ is not None:
            if agg == "mean":
                self.default_estimates_ = combined.groupby(self.default_category_, observed=True)["target"].mean()
            else:
                self.default_estimates_ = combined.groupby(self.default_category_, observed=True)["target"].median()
        else:
            self.default_estimates_ = None

        return self

    def predict(self, X_):
        """
        Predict target values for new observations based on group membership.
        Falls back to default_category estimate if combination missing (when configured).

        Parameters:
        X_ (pd.DataFrame or array-like): New observations to predict

        Returns:
        pd.Series: Predicted values, with NaN for entirely unknown groups
        """
        if self.group_estimates_ is None:
            raise ValueError("Model must be fitted before calling predict")

        # Normalize input to DataFrame and enforce column order
        if isinstance(X_, pd.DataFrame):
            X_pred = X_.copy()
            if set(X_pred.columns) != set(self.feature_names_):
                raise ValueError("Feature names in X_ do not match training data")
            X_pred = X_pred[self.feature_names_]
        else:
            X_pred = pd.DataFrame(X_, columns=self.feature_names_)

        # Build fast lookup dictionaries
        if isinstance(self.group_estimates_.index, pd.MultiIndex):
            full_map = {tuple(idx): float(val) for idx, val in self.group_estimates_.items()}
        else:
            # single index -> make keys be single-item tuples for uniformity
            full_map = {(idx,): float(val) for idx, val in self.group_estimates_.items()}

        default_map = None
        if self.default_estimates_ is not None:
            default_map = {k: float(v) for k, v in self.default_estimates_.items()}

        preds = []
        missing_count = 0
        for _, row in X_pred.iterrows():
            key = tuple(row[col] for col in self.feature_names_)
            val = full_map.get(key, None)
            if val is None and default_map is not None:
                # fallback using the default_category value
                default_val = default_map.get(row[self.default_category_], None)
                if default_val is not None:
                    val = default_val
            if val is None:
                val = np.nan
                missing_count += 1
            preds.append(val)

        if missing_count > 0:
            print(f"Warning: {missing_count} observations belong to groups not seen in training data")

        return pd.Series(preds, index=X_pred.index)

    def get_group_estimates(self):
        """Return the calculated group estimates (Series)."""
        return self.group_estimates_
# ...existing code...