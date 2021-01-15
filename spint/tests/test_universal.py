"""
Tests for universal spatial interaction models.
Test data is the Austria migration dataset used in Dennet's (2012) practical
primer on spatial interaction modeling. The data was made avialable through the
following dropbox link: http://dl.dropbox.com/u/8649795/AT_Austria.csv.
The data has been pre-filtered so that there are no intra-zonal flows.
Dennett, A. (2012). Estimating flows between geographical locations: get me
    started in spatial interaction modelling (Working Paper No. 184).
    UCL: Citeseer.
"""

__author__ = 'Tyler Hoffman tylerhoff1@gmail.com'

import unittest
import numpy as np
from scipy.stats import pearsonr
from ..universal import Lenormand, Radiation, PWO
np.random.seed(123456)


class TestUniversal(unittest.TestCase):
    """ Tests for universal models """

    def setUp(self):
        self.f = np.array([0,  1131,  1887,    69,   738,    98,
                          31,    43,    19, 1633,     0, 14055,   416,
                          1276,  1850,   388,   303,   159, 2301, 20164,
                          0,  1080,  1831,  1943,   742,   674,   407,
                          85,   379,  1597,     0,  1608,   328,   317,
                          469,   114, 762,  1110,  2973,  1252,     0,
                          1081,   622,   425,   262, 196,  2027,  3498, 346,
                          1332,     0,  2144,   821,   274, 49,   378,  1349,
                          310,   851,  2117,     0,   630,   106, 87,   424,
                          978,   490,   670,   577,   546,     0,   569,
                          33,   128,   643,   154,   328,   199, 112, 587, 0])

        self.o = np.array(['AT11', 'AT11', 'AT11', 'AT11', 'AT11', 'AT11',
                           'AT11', 'AT11', 'AT11', 'AT12', 'AT12', 'AT12',
                           'AT12', 'AT12', 'AT12', 'AT12', 'AT12', 'AT12',
                           'AT13', 'AT13', 'AT13', 'AT13', 'AT13', 'AT13',
                           'AT13', 'AT13', 'AT13', 'AT21', 'AT21', 'AT21',
                           'AT21', 'AT21', 'AT21', 'AT21', 'AT21', 'AT21',
                           'AT22', 'AT22', 'AT22', 'AT22', 'AT22', 'AT22',
                           'AT22', 'AT22', 'AT22', 'AT31', 'AT31', 'AT31',
                           'AT31', 'AT31', 'AT31', 'AT31', 'AT31', 'AT31',
                           'AT32', 'AT32', 'AT32', 'AT32', 'AT32', 'AT32',
                           'AT32', 'AT32', 'AT32', 'AT33', 'AT33', 'AT33',
                           'AT33', 'AT33', 'AT33', 'AT33', 'AT33', 'AT33',
                           'AT34', 'AT34', 'AT34', 'AT34', 'AT34', 'AT34',
                           'AT34', 'AT34', 'AT34'])

        self.d = np.array(['AT11', 'AT12', 'AT13', 'AT21', 'AT22', 'AT31',
                           'AT32', 'AT33', 'AT34', 'AT11', 'AT12', 'AT13',
                           'AT21', 'AT22', 'AT31', 'AT32', 'AT33', 'AT34',
                           'AT11', 'AT12', 'AT13', 'AT21', 'AT22', 'AT31',
                           'AT32', 'AT33', 'AT34', 'AT11', 'AT12', 'AT13',
                           'AT21', 'AT22', 'AT31', 'AT32', 'AT33', 'AT34',
                           'AT11', 'AT12', 'AT13', 'AT21', 'AT22', 'AT31',
                           'AT32', 'AT33', 'AT34', 'AT11', 'AT12', 'AT13',
                           'AT21', 'AT22', 'AT31', 'AT32', 'AT33', 'AT34',
                           'AT11', 'AT12', 'AT13', 'AT21', 'AT22', 'AT31',
                           'AT32', 'AT33', 'AT34', 'AT11', 'AT12', 'AT13',
                           'AT21', 'AT22', 'AT31', 'AT32', 'AT33', 'AT34',
                           'AT11', 'AT12', 'AT13', 'AT21', 'AT22', 'AT31',
                           'AT32', 'AT33', 'AT34'])

        self.dij = np.array([0, 103,  84, 221, 132, 215, 247, 391, 505,
                            103,   0,  46, 217, 130, 141, 201, 344, 454,
                            84,  46,   0, 250, 159, 186, 244, 288, 498,
                            221, 217, 250,   0,  92, 152,  93, 195, 306,
                            132, 130, 159, 92,   0, 125, 122, 262, 376,
                            215, 141, 186, 152, 125,   0,  82, 208, 315,
                            247, 201, 244,  93, 122,  82,   0, 145, 259,
                            391, 344, 388, 195, 262, 208, 145,   0, 114,
                            505, 454, 498, 306, 376, 315, 259, 114,   0])

        self.o_var = np.array([4016,  4016,  4016,  4016,  4016,  4016,
                              4016,  4016,  4016, 20080, 20080, 20080,
                              20080, 20080, 20080, 20080, 20080, 20080,
                              29142, 29142, 29142, 29142, 29142, 29142,
                              29142, 29142, 29142, 4897,  4897,  4897,
                              4897,  4897,  4897,  4897,  4897,  4897,
                              8487,  8487,  8487,  8487,  8487,  8487,
                              8487,  8487,  8487, 10638, 10638, 10638,
                              10638, 10638, 10638, 10638, 10638, 10638,
                              5790,  5790,  5790,  5790,  5790,  5790,
                              5790,  5790,  5790, 4341,  4341,  4341,
                              4341,  4341,  4341,  4341,  4341,  4341,
                              2184,  2184,  2184,  2184,  2184,  2184,
                              2184,  2184,  2184])

        self.d_var = np.array([5146, 25741, 26980,  4117,  8634,  8193,
                              4902,  3952,  1910, 5146, 25741, 26980,  4117,
                              8634,  8193,  4902,  3952,  1910, 5146, 25741,
                              26980,  4117,  8634,  8193,  4902,  3952,  1910,
                              5146, 25741, 26980,  4117,  8634,  8193,  4902,
                              3952,  1910, 5146, 25741, 26980,  4117,  8634,
                              8193,  4902,  3952,  1910, 5146, 25741, 26980,
                              4117,  8634,  8193,  4902,  3952,  1910,
                              5146, 25741, 26980,  4117,  8634,  8193,
                              4902,  3952,  1910, 5146, 25741, 26980,  4117,
                              8634,  8193,  4902,  3952,  1910, 5146, 25741,
                              26980,  4117,  8634,  8193,  4902,  3952,  1910])

        self.xlocs = np.array([47.1537, 48.1081, 48.2082, 46.7222, 47.3593,
                               48.0259, 47.8095, 47.2537, 47.2497])

        self.ylocs = np.array([16.2689, 15.805, 16.3738, 14.1806, 14.47,
                               13.9724, 13.055, 11.6015,  9.9797])

    def ready(self):
        N = 9
        outflows = self.o_var[0::N]
        inflows = self.d_var[0:N]
        locs = np.zeros((N, 2))
        locs[:, 0] = self.xlocs
        locs[:, 1] = self.ylocs
        dists = np.reshape(self.dij, (N, N), order='C')
        T_obs = np.reshape(self.f, (N, N), order='C')

        return outflows, inflows, locs, dists, T_obs

    def test_Lenormand(self):
        outflows, inflows, locs, dists, T_obs = self.ready()

        # Lenormand paper's model
        model = Lenormand(inflows, outflows, dists)
        T_L = model.flowmat()
        np.testing.assert_almost_equal(
            pearsonr(T_L.flatten(), T_obs.flatten()),
            (-0.07248415, 0.5181216)
        )

    def test_Radiation(self):
        outflows, inflows, locs, dists, T_obs = self.ready()

        # Radiation model -- requires locations of each node
        model = Radiation(inflows, outflows, dists, locs, locs)
        T_R = model.flowmat()
        np.testing.assert_almost_equal(
            pearsonr(T_R.flatten(), T_obs.flatten()),
            (0.05384603805950201, 0.6330568989373918)
        )

    def test_PWO(self):
        outflows, inflows, locs, dists, T_obs = self.ready()

        # PWO model
        model = PWO(inflows, outflows, dists, locs, locs)
        T_P = model.flowmat()
        np.testing.assert_almost_equal(
            pearsonr(T_P.flatten(), T_obs.flatten()),
            (0.23623562773229048, 0.033734908271368574)
        )


if __name__ == '__main__':
    unittest.main()
