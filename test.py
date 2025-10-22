# ...existing code...
import pandas as pd
import numpy as np
from apputil import GroupEstimate

def test_mean_basic():
    """Test mean estimation with basic input"""
    X = pd.DataFrame({'a': ['A','A','B'], 'b': ['X','Y','X']})
    y = [1, 2, 3]
    m = GroupEstimate('mean').fit(X, y)
    pred = m.predict(X)
    assert list(pred) == [1.0, 2.0, 3.0]

def test_median_basic():
    """Test median estimation with basic input"""
    X = pd.DataFrame({'a': ['A','A','A'], 'b': ['X','X','X']})
    y = [1, 3, 2]
    m = GroupEstimate('median').fit(X, y)
    pred = m.predict(X)
    assert list(pred) == [2.0, 2.0, 2.0]

def test_unknown_groups_prints_warning(capsys=None):
    """Test that predicting on unknown groups prints a warning and returns NaN"""
    X_train = pd.DataFrame({'a': ['A'], 'b': ['X']})
    y = [1]
    m = GroupEstimate('mean').fit(X_train, y)
    X_test = pd.DataFrame({'a': ['A', 'B'], 'b': ['X', 'Y']})
    if capsys is not None:
        # pytest will supply capsys fixture
        pred = m.predict(X_test)
        captured = capsys.readouterr()
        out = captured.out
    else:
        # running as a plain script: capture stdout manually
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pred = m.predict(X_test)
        out = buf.getvalue()
    assert pd.isna(pred.iloc[1])
    assert "Warning" in out

if __name__ == "__main__":
    for fn in (test_mean_basic, test_median_basic, test_unknown_groups_prints_warning):
        try:
            fn()
            print(f"{fn.__name__}: OK")
        except AssertionError:
            print(f"{fn.__name__}: FAILED")
            raise
# ...existing code...