import pandas as pd
import pytest

from qm.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["paths"] = {"data_dir": tmp_path / "data", "results_dir": tmp_path / "results"}
    return c


def frame(dates, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates))
