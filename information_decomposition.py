"""Utilities for computing a positive information decomposition for
distributions over three binary variables using the definitions from
Bertschinger at al (2014)."""
import numpy as np

# basis vectors for the optimization domain
gamma_0 = np.array([1, -1, -1, 1, 0, 0, 0, 0], dtype=np.float)
gamma_1 = np.array([0, 0, 0, 0, 1, -1, -1, 1], dtype=np.float)

def I_XYZ(P):
    PXYZ = P.reshape((2,2,2))
    PX = np.einsum('ijk->i', PXYZ).reshape((2,1,1))
    PYZ = np.einsum('ijk->jk', PXYZ).reshape((1,2,2))
    return (PXYZ * np.log2(PXYZ / (PX * PYZ)))[PXYZ>0].sum()

def I_2D(Q):
    QX = np.atleast_2d(Q.sum(axis=1))
    QY = np.atleast_2d(Q.sum(axis=0))
    if min(QX.min(), QY.min()) < 0 or Q.min()<0:
        raise Exception
    table = Q * np.log2(Q / (QX.transpose() * QY))
    return table[Q>0].sum()

def I_Q_XY(Q, axis=2):
    QXY = Q.reshape((2,2,2)).sum(axis=axis)
    return I_2D(QXY)

def I_Q_XY_Z(Q):
    Q_XYZ = Q.reshape((2,2,2))
    return Q_XYZ[:,:,0].sum() * I_2D(Q_XYZ[:,:,0]/Q_XYZ[:,:,0].sum()) + Q_XYZ[:,:,1].sum() * I_2D(Q_XYZ[:,:,1]/Q_XYZ[:,:,1].sum())

def CoI_Q(P, a, b):
    # co-information CoI_Q(C;fUr;fUo)
    Q = P + a * gamma_0 + b * gamma_1
    return I_Q_XY(Q) - I_Q_XY_Z(Q)

def parameter_bounds(P):
    a_upper = min(min(P[1], P[2]), 1-max(P[0], P[3]))
    a_lower = -min(1-max(P[1],P[2]), min(P[0], P[3]))
    b_upper = min(min(P[5], P[6]), 1-max(P[4], P[7]))
    b_lower = -min(1-max(P[5],P[6]), min(P[4], P[7]))
    return a_lower, a_upper, b_lower, b_upper

def parameter_meshgrid(P):
    a_lower, a_upper, b_lower, b_upper = parameter_bounds(P)
    a_range = np.linspace(a_lower, a_upper, 100)
    b_range = np.linspace(b_lower, b_upper, 100)
    AA, BB = np.meshgrid(a_range, b_range)
    return AA, BB

def coinformation_table(P):
    AA, BB = parameter_meshgrid(P)
    CoIQQ = np.zeros_like(AA)
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            CoIQQ[i,j] = CoI_Q(P, AA[i,j], BB[i,j])
    return CoIQQ

def decomposition(P):
    SI = coinformation_table(P).max()
    UI_Y = I_Q_XY(P, axis=2) - SI
    UI_Z = I_Q_XY(P, axis=1) - SI
    CI = I_XYZ(P) - SI - UI_Y - UI_Z
    return SI, CI, UI_Y, UI_Z
    
