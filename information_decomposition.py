"""Utilities for computing a positive information decomposition for
distributions over three binary variables using the definitions from
Bertschinger at al (2014)."""
import numpy as np

# GAMMA_0 and GAMMA_1 are the (constant) basis vectors for the
# optimization domain $\Delta_P$. In the case of three binary
# variables, $\Delta_P$ has dimension 2. We express the vectors in the
# same "flattened joint probability distribution" coordinate system we
# will use below for P. So, for instance, GAMMA_0 corresponds to both
# gamma_(00101) and gamma_(01010) (which are identical) in the
# notation of the Bertschinger paper. All other gamma_(0....) vectors
# are zero. Since, by direct computation from the definition,
#
# gamma_(00101)=d_000+d_011-d_010-d_001
#
# where d_xyz is the Kronecker delta, we get that in our chosen
# coordinate system for the space of joint PDFs over (x,y,z) the
# representation of gamma_(00101) is the one given below. An analogous
# argument leads to the expression for GAMMA_1.
GAMMA_0 = np.array([1, -1, -1, 1, 0, 0, 0, 0], dtype=np.float) # this 
GAMMA_1 = np.array([0, 0, 0, 0, 1, -1, -1, 1], dtype=np.float)

def I_XYZ(P):
    """Compute mutual information I(X:(Y,Z)) from P(X,Y,Z), where P can be
    specified as a flattened joint probability table.

    """
    PXYZ = P.reshape((2,2,2))
    PX = np.einsum('ijk->i', PXYZ).reshape((2,1,1))
    PYZ = np.einsum('ijk->jk', PXYZ).reshape((1,2,2))
    with np.errstate(divide='ignore', invalid='ignore'):
        # division by zero and such warnings can be safely ignored as
        # at the end we're only summing over those values where
        # P(x,Y,Z) is strictly positive
        result = (PXYZ * np.log2(PXYZ / (PX * PYZ)))[PXYZ>0].sum()
    return result

def I_2D(Q):
    """Compute mutual information I(X:Y) from Q(X,Y), where Q must be
    specified as a XxY joiint probability table.

    """
    QX = np.atleast_2d(Q.sum(axis=1))
    QY = np.atleast_2d(Q.sum(axis=0))
    if min(QX.min(), QY.min()) < 0 or Q.min()<0:
        raise Exception
    with np.errstate(divide='ignore', invalid='ignore'):
        table = Q * np.log2(Q / (QX.transpose() * QY))
    return table[Q>0].sum()

def I_Q_XY(Q, axis=2):
    """Compute mutual information I(X:Y) from Q(X,Y,Z), expressed as a
    flattened joint probability table.

    """
    QXY = Q.reshape((2,2,2)).sum(axis=axis)
    return I_2D(QXY)

def I_Q_XY_Z(Q):
    """Compute mutual information I(X:Y|Z) from Q(X,Y,Z), expressed as a
    flattened joint probability table.

    """
    Q_XYZ = Q.reshape((2,2,2))
    return Q_XYZ[:,:,0].sum() * I_2D(Q_XYZ[:,:,0]/Q_XYZ[:,:,0].sum()) + Q_XYZ[:,:,1].sum() * I_2D(Q_XYZ[:,:,1]/Q_XYZ[:,:,1].sum())

def CoI_Q(P, a, b):
    """Compute the co-information CoI_Q(X;Y;Z) of the probability
    distribution Q, belonging to $\Delta_P$ and generated from P and a
    specific choice of the value of the coordinates (a,b).

    """
    Q = P + a * GAMMA_0 + b * GAMMA_1
    return I_Q_XY(Q) - I_Q_XY_Z(Q)

def domain_boundaries(P):
    """Compute boundaries for the polytope $\Delta_P$, which is the
    optimisation domain we need to consider in the computation of the
    positive information decomposition. Since Q=P+a*GAMMA_0+b*GAMMA_1
    for some value of a and b for each Q in $\Delta_P$, the boundaries
    of $\Delta_P$ are given by the requirement that Q must always be a
    valid probability distribution. To understand how the expressions
    below are derived, it is sufficient to write down (for instance)
    P+a*GAMMA_0 explicitly in terms of coordinates, yielding

    [p_000+a, p_001-a, p_010-a, p_011+a, p_100, p_101, p_110, p_111]

    and to notice that, if this is to be a valid pdf, a must be larger
    than -p_000, smaller than 1-p_000, and so on.

    """
    a_upper = min(min(P[1], P[2]), 1-max(P[0], P[3]))
    a_lower = -min(1-max(P[1],P[2]), min(P[0], P[3]))
    b_upper = min(min(P[5], P[6]), 1-max(P[4], P[7]))
    b_lower = -min(1-max(P[5],P[6]), min(P[4], P[7]))
    return a_lower, a_upper, b_lower, b_upper

def parameter_meshgrid(P, npoints):
    """Prepare a meshgrid of combinations of the coordinates a and b,
    parametrising the domain $\Delta_P$ over which we want to find the
    probability distribution with the maximum coinformation.

    """
    a_lower, a_upper, b_lower, b_upper = domain_boundaries(P)
    a_range = np.linspace(a_lower, a_upper, npoints)
    b_range = np.linspace(b_lower, b_upper, npoints)
    AA, BB = np.meshgrid(a_range, b_range)
    return AA, BB

def coinformation_table(P, npoints):
    """Compute the coinformation for a number of probability distributions
    situated on a grid in $\Delta_P$.

    Parameters
    ----------
    P: the probability distribution from which to generate the domain $\Delta_P$.

    npoints: linear resolution of the grid. The total number of points
    will be npoints^2.

    """
    AA, BB = parameter_meshgrid(P, npoints)
    CoIQQ = np.zeros_like(AA)
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            CoIQQ[i,j] = CoI_Q(P, AA[i,j], BB[i,j])
    return CoIQQ

def decomposition(P, npoints=100):
    """Positive information decomposition of distribution over three binary variables.

    This is a toy implementation of the decomposition defined in
    Bertschinger et al (2014).

    Parameters
    ----------
    P : 2x2x2 numpy array.
    The joint probability table of the distribution. Each entry must be a
    probability, so 0<=P[i,j,k]<=1 for all i,j,k and P.sum()=1.

    Probabilities are specified indexing into the table by their value:
    
    p(x=0,y=0,z=0) = P[0,0,0]
    p(x=0,y=0,z=1) = P[0,0,1]
    
    ...and so on.

    npoints: the number of points to consider on each side of the mesh
    grid we will use to optimise the coinformation over $\Delta_P$.

    Examples
    --------
    >>> # PID for y and z being uniformly distributed and x=AND(y,z):
    >>> import numpy as np
    >>> from information_decomposition import decomposition
    >>> P = np.array([0.25, 0.25, 0.25, 0, 0, 0, 0, 0.25]).reshape(2,2,2)
    >>> SI, CI, UI_Y, UI_Z = decomposition(P)
    >>> SI
    0.31127812445913278
    >>> CI
    0.49999999999999994
    >>> UI_Y
    0.0
    >>> UI_Z
    0.0

    """
    # make sure P is correctly specified as a joint probability table
    # over binary variables
    P = P.astype(np.float) 
    assert P.shape == (2,2,2)
    assert all(np.logical_and(P.flat>=0, P.flat<=1)) and P.sum()==1

    P = P.flat[:]
    # shared information SI is the maximum of the coinformation over a 
    SI = coinformation_table(P, npoints).max()
    UI_Y = I_Q_XY(P, axis=2) - SI
    UI_Z = I_Q_XY(P, axis=1) - SI
    CI = I_XYZ(P) - SI - UI_Y - UI_Z
    return SI, CI, UI_Y, UI_Z
    
