"""Utilities for the application of Bertschinger's positive
information decomposition to the analysis of intersection information
in sensory-motor decision making.
"""
import numpy as np

import information_decomposition as pid

def compute_shuffle(P):
    """Compute distribution to use as a null model for the computation of
    intersection. This is the distribution with the same (S,R) and
    (S,C) marginals as the real data, but whose graphical structure is
    R<-S->C.

    """
    PX = np.einsum('ijk->i', P).reshape((2,1))
    PXY = np.einsum('ijk->ij', P)
    PXZ = np.einsum('ijk->ik', P)
    return np.einsum('ij,ik->ijk', PXY, PXZ)/PX


def intersection_information(P, npoints=100):
    """Compute intersection information from P(S,R,C).

    Intersection information is computed as SI(R:S;C)-SI_n(R:S;C),
    where the first term is the shared information (according to
    Bertschinger) between S and C about R, and the second term is the
    same quantity computed for a "null" distribution where the (S,R)
    and (S,C) marginals are the same but the graphical structure is
    R<-S->C (i.e. shuffled data without signal correlations). The idea
    behind this definition of null distribution is to subtract out the
    maximum amount of shared information that could be due to signal
    correlations.

    Parameters
    ----------
    P : 2x2x2 numpy array.
    Joint (S,R,C) probability table, specified indexing into the table
    by the values of S, R and C as for the 'decomposition' function.

    npoints: see the documentation for 'decomposition'.


    Returns
    -------
    II : intersection information for (S,R,C)
    SI_P : intersection information before subtraction of null value
    SI_Pn : intersection information for null distribution

    """
    # compute shared information for real data. Note that
    # pid.decomposition uses the variable linked to the first index of
    # P as "output" variable, so if P is indexed by (S,R,C) in this
    # order we have to permute the array describing the joint pdf
    # before passing it to the function.
    SI_P = pid.decomposition(np.transpose(P, (1,0,2)), npoints=npoints)[0]

    # compute null distribution
    Pn = compute_shuffle(P)

    # compute shared information for null distribution
    SI_Pn = pid.decomposition(np.transpose(Pn, (1,0,2)), npoints=npoints)[0]

    return SI_P - SI_Pn, SI_P, SI_Pn
    
    
