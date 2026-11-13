from types import SimpleNamespace

import numpy as np
import pytest

from spint.utils import CPC


def test_CPC_perfect_fit():
    y = np.array([10.0, 0.0, 5.0, 25.0])
    model = SimpleNamespace(y=y, yhat=y.copy())
    assert pytest.approx(CPC(model)) == 1.0


def test_CPC_no_common_flows():
    model = SimpleNamespace(y=np.array([4.0, 0.0]), yhat=np.array([0.0, 6.0]))
    assert pytest.approx(CPC(model)) == 0.0


def test_CPC_partial_fit():
    model = SimpleNamespace(y=np.array([10.0, 20.0]), yhat=np.array([15.0, 15.0]))
    # 2 * (10 + 15) / (30 + 30)
    assert pytest.approx(CPC(model)) == 50.0 / 60.0
