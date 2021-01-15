"""
Implementations of universal spatial interaction models: Lenormand's
model, radiation model, and population-weighted opportunities.
References
----------
Lenormand, M., Huet, S., Gargiulo, F., and Deffuant, G. (2012). "A Universal
    Model of Commuting Networks." PLOS One, 7, 10.
Simini, F., Gonzalez, M. C., Maritan, A., Barabasi, A.-L. (2012). "A universal
    model for mobility and migration patterns." Nature, 484, 96-100.
Yan, X.-Y., Zhao, C., Fan, Y., Di, Z., and Wang, W.-X. (2014). "Universal
    predictability of mobility patterns in cities." Journal of the Royal
    Society Interface, 11, 100.
"""

__author__ = 'Tyler Hoffman tylerhoff1@gmail.com'

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


class Universal(ABC):
    """
    Base class for all the universal models as they all have similar
    underlying structures. For backend design purposes, not practical use.
    Parameters
    ----------
    inflows         : array of reals
                      N x 1, observed flows into each location
    outflows        : array of reals
                      M x 1, observed flows out of each location
    dists           : matrix of reals
                      N x M, pairwise distances between each location
    Attributes
    ----------
    N               : integer
                      number of origins
    M               : integer
                      number of destinations
    flowmat         : abstract method
                      estimates flows, implemented by children
    """
    def __init__(self, inflows, outflows, dists):
        self.N = len(outflows)           # number of origins
        self.M = len(inflows)            # number of destinations
        self.outflows = outflows.copy()  # list of origin outflows
        self.inflows = inflows.copy()    # list of destination inflows
        self.dists = dists.copy()        # list of distances

    @abstractmethod
    def flowmat(self): pass


class Lenormand(Universal):
    """
    Universal model based off of Lenormand et al. 2012,
    "A Universal Model of Commuting Networks".
    Parameters
    ----------
    inflows         : array of reals
                      N x 1, observed flows into each location
    outflows        : array of reals
                      M x 1, observed flows out of each location
    dists           : matrix of reals
                      N x M, pairwise distances between each location
    beta            : scalar
                      real, universal parameter for the model
    avg_sa          : scalar
                      real, average surface area of units
    Attributes
    ----------
    N               : integer
                      number of origins
    M               : integer
                      number of destinations
    calibrate       : method
                      calibrates beta using constants from the paper
    flowmat         : method
                      estimates flows via the Lenormand model
    """

    def __init__(self, inflows, outflows, dists, beta=1, avg_sa=None):
        super().__init__(inflows, outflows, dists)
        self.beta = self.calibrate(avg_sa) if avg_sa is not None else beta

    def calibrate(self, avg_sa):
        # Constants from the paper
        nu = 0.177
        alpha = 3.15 * 10**(-4)
        self.beta = alpha*avg_sa**(-nu)

    def flowmat(self):
        # Builds the matrix T from the parameter beta and a matrix of distances
        T = np.zeros((self.N, self.M))

        # Copy class variables so as not to modify
        sIN = self.inflows.copy()
        sOUT = self.outflows.copy()

        # Assembly loop
        while sum(sOUT) > 0:
            # Pick random nonzero sOUT
            idxs, = np.where(sOUT > 0)
            i = np.random.choice(idxs)

            # Compute Pij's (not memoized b/c it changes on iteration)
            Pi = np.multiply(sIN, np.exp(-self.beta*self.dists[i, :])) / \
                np.dot(sIN, np.exp(-self.beta*self.dists[i, :]))

            # Pick random j according to Pij
            j = np.random.choice(range(self.N), p=Pi)

            # Adjust values
            T[i, j] += 1
            sIN[j] -= 1
            sOUT[i] -= 1

        return T


class Radiation(Universal):
    """
    Universal model based off of Simini et al. 2012,
    "A universal model for mobility and migration patterns".
    Requires slightly more data than Lenormand.
    Parameters
    ----------
    inflows         : array of reals
                      N x 1, observed flows into each location
    outflows        : array of reals
                      M x 1, observed flows out of each location
    dists           : matrix of reals
                      N x M, pairwise distances between each location
    ilocs           : array of reals
                      N x 2, inflow node locations
    olocs           : array of reals
                      M x 2, outflow node locations
    Attributes
    ----------
    N               : integer
                      number of origins
    M               : integer
                      number of destinations
    flowmat         : method
                      estimates flows via the Radiation model
    """

    def __init__(self, inflows, outflows, dists, ilocs, olocs):
        super().__init__(inflows, outflows, dists)
        self.ilocs = ilocs.copy()
        self.olocs = olocs.copy()

    def _from_origin(self, idx, total_origins):
        # Sort destinations by distance from origin
        didxs = np.argsort(self.dists[idx, :])
        inflows = self.inflows[didxs]

        # Normalization
        F = 1.0/(1.0 - self.outflows[idx]/total_origins)

        pop_in_radius = 0
        flows = np.zeros((self.M,))
        for j in range(self.M):
            # Use formula from the paper
            flows[j] = F*(self.outflows[idx]*inflows[j]) / \
                       ((self.outflows[idx] + pop_in_radius) *
                        (self.outflows[idx] + inflows[j] + pop_in_radius))

            pop_in_radius += inflows[j]

        # Unsort list
        return flows[didxs.argsort()]

    def flowmat(self):
        # Builds the OD matrix T from the inputted data
        T = np.zeros((self.N, self.M))
        total_origins = sum(self.outflows)

        for i in range(self.N):
            T[i, :] = self._from_origin(i, total_origins)

        return T


class PWO(Universal):
    """
    Population-weighted opportunies (PWO) implements a
    universal model based off of Yan et al. 2014,
    "Universal predictability of mobility patterns in cities".
    Requires slightly more data than Lenormand.
    Parameters
    ----------
    inflows         : array of reals
                      N x 1, observed flows into each location
    outflows        : array of reals
                      M x 1, observed flows out of each location
    dists           : matrix of reals
                      N x M, pairwise distances between each location
    ilocs           : array of reals
                      N x 2, inflow node locations
    olocs           : array of reals
                      M x 2, outflow node locations
    Attributes
    ----------
    N               : integer
                      number of origins
    M               : integer
                      number of destinations
    flowmat         : method
                      estimates flows via the Radiation model
    """

    def __init__(self, inflows, outflows, dists, ilocs, olocs):
        super().__init__(inflows, outflows, dists)
        self.ilocs = ilocs.copy()
        self.olocs = olocs.copy()
        self.total = sum(inflows)  # total population of the system

    def _from_destination(self, jdx):
        # Sort origins by distance from destination
        didxs = np.argsort(self.dists[jdx, :])
        outflows = self.outflows[didxs]
        pop_in_radius = self.inflows[jdx]  # here pop_in_radius includes endpts
        flows = np.zeros((self.N,))

        # Loop over origins
        for i in range(self.N):
            pop_in_radius += outflows[i]  # add other endpt

            # Compute denominator
            denom = 0
            denom_pop_in_radius = outflows[i]
            for k in range(self.M):  # loop over destinations
                denom_pop_in_radius += self.inflows[k]
                if k != i:
                    denom += self.inflows[k] * (1/denom_pop_in_radius -
                                                1/self.total)

            # Use formula from the paper
            flows[i] = self.inflows[jdx]*(1/pop_in_radius - 1/self.total)/denom

        # Unsort list
        return flows[didxs.argsort()]

    def flowmat(self):
        # Builds the OD matrix T from the inputted data
        T = np.zeros((self.N, self.M))

        for j in range(self.M):
            T[:, j] = self._from_destination(j)

        return T


def test():
    # Read data from Austria file
    N = 9
    austria = pd.read_csv('austria.csv')
    modN = austria[austria.index % N == 0]
    outflows = modN['Oi'].values
    inflows = austria['Dj'].head(n=N).values
    locs = np.zeros((N, 2))
    locs[:, 0] = modN['X'].values
    locs[:, 1] = modN['Y'].values
    dists = np.reshape(austria['Dij'].values, (N, N), order='C')
    T_obs = np.reshape(austria['Data'].values, (N, N), order='C')

    # Lenormand paper's model
    model = Lenormand(inflows, outflows, dists)
    T_L = model.flowmat()
    print(pearsonr(T_L.flatten(), T_obs.flatten()))

    # Radiation model -- requires locations of each node
    model = Radiation(inflows, outflows, dists, locs, locs)
    T_R = model.flowmat()
    print(pearsonr(T_R.flatten(), T_obs.flatten()))

    # PWO model
    model = PWO(inflows, outflows, dists, locs, locs)
    T_P = model.flowmat()
    print(pearsonr(T_P.flatten(), T_obs.flatten()))


if __name__ == '__main__':
    test()
