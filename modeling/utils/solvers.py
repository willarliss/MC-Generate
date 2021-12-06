"""Eigen and singular value solver functions"""

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

import numpy as np
import scipy.sparse as sp


def eigen_solver(A, k=None, random_state=None):

    shape = A.shape[0]

    return sp.linalg.eigs(
        A=A,
        tol=0.,
        which='LM',
        maxiter=shape*10,
        k=(int(np.log(shape)) if k is None else int(k)),
        v0=np.random.default_rng(random_state).normal(size=shape),
    )


def singular_value_solver(A, k=None, random_state=None):

    shape = A.shape[0]

    return sp.linalg.svds(
        A=A,
        tol=0.,
        which='LM',
        solver='arpack',
        maxiter=shape*10,
        k=(int(np.log(shape)) if k is None else int(k)),
        v0=np.random.default_rng(random_state).normal(size=shape),
    )
