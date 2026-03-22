"""
Tests for analysis of spatial autocorrelation within vectors

"""

__author__ = "Taylor Oshan tayoshan@gmail.com"


import numpy as np
import pytest
from libpysal.weights.distance import DistanceBand

from ..vec_SA import VecMoran


class TestVecMoran:
    """Tests VecMoran class"""

    def setup_method(self):
        self.vecs = np.array(
            [
                [1, 55, 60, 100, 500],
                [2, 60, 55, 105, 501],
                [3, 500, 55, 155, 500],
                [4, 505, 60, 160, 500],
                [5, 105, 950, 105, 500],
                [6, 155, 950, 155, 499],
            ]
        )
        self.origins = self.vecs[:, 1:3]
        self.dests = self.vecs[:, 3:5]

    def test_origin_focused_A(self):
        wo = DistanceBand(self.origins, threshold=9999, alpha=-1.5, binary=False)
        np.random.seed(1)
        vmo = VecMoran(self.vecs, wo, focus="origin", rand="A")
        assert pytest.approx(vmo.I) == 0.645944594367
        assert pytest.approx(vmo.p_z_sim) == 0.03898650733809228

    def test_dest_focused_A(self):
        wd = DistanceBand(self.dests, threshold=9999, alpha=-1.5, binary=False)
        np.random.seed(1)
        vmd = VecMoran(self.vecs, wd, focus="destination", rand="A")
        assert pytest.approx(vmd.I) == -0.764603695022
        assert pytest.approx(vmd.p_z_sim) == 0.149472673677

    def test_origin_focused_B(self):
        wo = DistanceBand(self.origins, threshold=9999, alpha=-1.5, binary=False)
        np.random.seed(1)
        vmo = VecMoran(self.vecs, wo, focus="origin", rand="B")
        assert pytest.approx(vmo.I) == 0.645944594367
        assert pytest.approx(vmo.p_z_sim) == 0.02944612633233532

    def test_dest_focused_B(self):
        wd = DistanceBand(self.dests, threshold=9999, alpha=-1.5, binary=False)
        np.random.seed(1)
        vmd = VecMoran(self.vecs, wd, focus="destination", rand="B")
        assert pytest.approx(vmd.I) == -0.764603695022
        assert pytest.approx(vmd.p_z_sim) == 0.12411761124197379
