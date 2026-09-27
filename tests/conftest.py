import pandas as pd
import pytest

from qm.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["paths"] = {"data_dir": tmp_path / "data", "results_dir": tmp_path / "results"}
    c["dates"]["snapshot_end"] = None  # tests never depend on the real frozen date
    return c


def frame(dates, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates))
