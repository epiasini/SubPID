import numpy as np
import cvxpy

def I_2D(q_xy):
    """Compute mutual information I(X:Y) from Q(X,Y), where Q must be
    specified as a XxY joint probability table.

    """
    q_x = q_xy.sum(axis=1, keepdims=True)
    q_y = q_xy.sum(axis=0, keepdims=True)
    if min(q_x.min(), q_y.min()) < 0 or q_xy.min()<0:
        raise Exception
    with np.errstate(divide='ignore', invalid='ignore'):
        table = q_xy * np.log2(q_xy / (q_x * q_y))
    return table[q_xy>0].sum()

def I_3D_cond_z(q_xyz):
    """Compute conditional mutual information I(X:Y|Z) from Q(X,Y,Z),
    where Q must be specified as a dimx*dimy*dimz 3D joint probability
    table.
    """
    q_z = q_xyz.sum(axis=(1,2), keepdims=True)
    q_xz = q_xyz.sum(axis=1, keepdims=True)
    q_yz = q_xyz.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        q_x_z = q_xz / q_z
        q_y_z = q_yz / q_z
        q_xy_z = q_xyz / q_z
        table = q_xyz * np.log2(q_xy_z/(q_x_z*q_y_z))
    table[np.logical_not(np.isfinite(table))] = 0
    return table.sum()    

def lin_pid(p, accuracy=0.01, method='glpk', verbose=False):

    # work out dimensionality of the input probability distribution
    (dimx, dimy, dimz) = p.shape

    # Following Bertschinger2013, define a set of Gamma matrices that
    # forms a basis of the optimization domain. All Gammas have the
    # same dimensions as the input p. Their total number
    # (dimx-1)*(dimy-1)*dimz (See Bertschinger2013, Lemma 26). For
    # ease of manipulation, we keep them all stacked into a larger
    # array.
    GAMMA = np.zeros((dimx, dimy, dimz, dimx-1, dimy-1, dimz))
    for iz in range(dimz):
        for ix in range(dimx-1):
            for iy in range(dimy-1):
                GAMMA[ix,iy,iz,ix,iy,iz] = 1;
                GAMMA[ix+1,iy,iz,ix,iy,iz] = -1;
                GAMMA[ix,iy+1,iz,ix,iy,iz] = -1;
                GAMMA[ix+1,iy+1,iz,ix,iy,iz] = 1;

    # Algorithm termination flag
    done = False;

    # numbers of iterations of the algorithm
    iteration = 0

    # Coefficients of the matrix q expressed in the basis of Gamma
    # matrices
    parameters = np.zeros((dimz*(dimx-1)*(dimy-1), 1))

    # starting point of the algorithm
    q = p[:]
    coeff_prev = parameters[:]

    

 
    while not done:
        # compute derivative of MI_q along the basis vectors
        with np.errstate(divide='ignore', invalid='ignore'):
            deriv = np.log2(q[:-1,:-1]*q[1:,1:]) - \
                    np.log2(q[:-1,1:]*q[1:,:-1]) + \
                    np.log2(q[:-1,1:].sum(axis=2, keepdims=True) * \
                            q[1:,:-1].sum(axis=2, keepdims=True)) - \
                    np.log2(q[:-1,:-1].sum(axis=2, keepdims=True) * \
                            q[1:,1:].sum(axis=2, keepdims=True))

        # get rid of nonsense values of deriv coming from finite numerical precision
        deriv[np.logical_not(np.isfinite(deriv))] = 0
    
        # stores the coefficients of the q of the current
        # iteration. For each value of z there is a coeff, then
        # coeff_tot stacks all coeffs.
        coeff_tot = np.zeros(((dimx-1)*(dimy-1), dimz))

        for iz in range(dimz):
            
            # the constraints on q, for each z, are implemented via the inequality A*coeff<=b
            b = np.zeros((2*dimx*dimy,1))
            A = np.zeros((2*dimx*dimy, (dimx-1)*(dimy-1)))

            # the number of constraints to be imposed
            count = 0

            for ix in range(dimx-1):
                for iy in range(dimy-1):
                    if ix==0 and iy==0:
                        # set constraints for the top-left entries q[0,0,iz]
                        A[count, ix*(dimy-1)+iy] = -1
                        A[count+dimx*dimy, ix*(dimy-1)+iy] = 1
                        b[count] = p[ix,iy,iz]
                        b[count+dimx*dimy] = 1-p[ix,iy,iz]
                        count += 1
            
                    if ix==dimx-2 and iy==dimy-2:
                        # bottom right entries
                        A[count, ix*(dimy-1)+iy] = -1
                        A[count+dimx*dimy, ix*(dimy-1)+iy] = 1
                        b[count] = p[ix+1,iy+1,iz]
                        b[count+dimx*dimy] = 1-p[ix+1,iy+1,iz]
                        count += 1
                        
                    if ix==0 and iy==dimy-2:
                        # top right entries
                        A[count, ix*(dimy-1)+iy] = 1
                        A[count+dimx*dimy, ix*(dimy-1)+iy]=-1
                        b[count] = p[ix,iy+1,iz]
                        b[count+dimx*dimy] = 1-p[ix,iy+1,iz]
                        count += 1
                        
                    if ix==dimx-2 and iy==0:
                        # bottom left entries
                        A[count, ix*(dimy-1)+iy] = 1
                        A[count+dimx*dimy, ix*(dimy-1)+iy] = -1
                        b[count] = p[ix+1,iy,iz]
                        b[count+dimx*dimy] = 1-p[ix+1,iy,iz]
                        count += 1

                    if ix==0 and dimy>2 and iy<dimy-2:
                        # top row entries, from left to right
                        A[count, ix*(dimy-1)+iy] = 1
                        A[count, ix*(dimy-1)+iy+1] = -1
                        A[count+dimx*dimy, ix*(dimy-1)+iy] = -1
                        A[count+dimx*dimy, ix*(dimy-1)+iy+1] = 1
                        b[count] = p[ix,iy+1,iz]
                        b[count+dimx*dimy] = 1-p[ix,iy+1,iz]
                        count += 1

                    if iy==0 and dimx>2 and ix<dimx-2:
                        # left-most column, top-down
                        A[count, ix*(dimy-1)+iy]=1
                        A[count, (ix+1)*(dimy-1)+iy]=-1
                        A[count+dimx*dimy, ix*(dimy-1)+iy] = -1
                        A[count+dimx*dimy, (ix+1)*(dimy-1)+iy] = 1
                        b[count] = p[ix+1,iy,iz]
                        b[count+dimx*dimy] = 1-p[ix+1,iy,iz]
                        count += 1

                    if iy==dimy-2 and dimx>2 and ix<dimx-2:
                        # right-most column, top-down
                        A[count, ix*(dimy-1)+iy] = -1
                        A[count, (ix+1)*(dimy-1)+iy] = 1
                        A[count+dimx*dimy, ix*(dimy-1)+iy] = 1
                        A[count+dimx*dimy, (ix+1)*(dimy-1)+iy] = -1
                        b[count] = p[ix+1,iy+1,iz]
                        b[count+dimx*dimy] = 1-p[ix+1,iy+1,iz]
                        count +=1

                    if dimx>2 and dimy>2 and 0<ix<=dimx-2 and 0<iy<=dimy-2:
                        # internal
                        A[count, ix*(dimy-1)+iy] = -1
                        A[count, (ix-1)*(dimy-1)+iy] = 1
                        A[count, ix*(dimy-1)+iy-1] = 1;
                        A[count, (ix-1)*(dimy-1)+iy-1] = -1;
                        
                        A[count+dimx*dimy, ix*(dimy-1)+iy] = 1
                        A[count+dimx*dimy, (ix-1)*(dimy-1)+iy] = -1
                        A[count+dimx*dimy, ix*(dimy-1)+iy-1] = -1
                        A[count+dimx*dimy, (ix-1)*(dimy-1)+iy-1] = 1

                        b[count] = p[ix,iy,iz]
                        b[count+dimx*dimy] = 1-p[ix,iy,iz]
                        count += 1

            # extract the portion of deriv pertaining to iz and adapt
            # deriv_iz to the multiplication deriv_iz*coeff
            deriv_iz = deriv[:,:,iz].reshape((-1,1), order='C')

            if method=='cvx':
                raise NotImplementedError('CVX not yet implemented!')
            elif method=='glpk':
                x = cvxpy.Variable(deriv_iz.size)
                obj = cvxpy.Minimize(deriv_iz.T*x)
                constraints = [A*x<=b]
                prob = cvxpy.Problem(obj, constraints)
                prob.solve(verbose=verbose)

            coeff_tot[:,iz] = prob.value

        coeff_tot = coeff_tot.reshape((-1,1), order='F')


        p_k = p[:]
        for ind_gamma in range((dimx-1)*(dimy-1)*dimz):
            ix = np.mod(ind_gamma/(dimy-1), dimx-1).astype(np.int)
            iy = np.mod(ind_gamma, dimy-1).astype(np.int)
            iz = np.floor(ind_gamma/((dimx-1)*(dimy-1))).astype(np.int)
            p_k = p_k + coeff_tot[ind_gamma]*GAMMA[:,:,:,ix,iy,iz]
      
        # this is supposed to build a concatenated array where the
        # forst block corresponds to deriv_iz for iz=0, the second
        # corresponds to deriv_iz for iz=1, etc
        deriv = np.transpose(deriv, axes=(1,0,2)).reshape((-1,1), order='F')
        
        #set the stopping criterion based on the duality gap, see Stratos;
        if iteration>100:#iteration>0 and np.dot(deriv.T,coeff_prev-coeff_tot)<=accuracy:
            print(iteration)
            done = True # exit the algorithm
        else:
            # fixed increment; simplest version of Franke-Wolf
            gamma_k = 2/(iteration+2)
            # update the q for next iteration
            q = q + gamma_k*(p_k-q)
            # update coeff_prev for next i
            coeff_prev = coeff_prev + gamma_k*(coeff_tot-coeff_prev)

        iteration += 1
            
        # get rid of tiny negative entries in q which result from
        # limited numerical precision
        q[q<0] = 0
        


    # compute co-information for optimal distribution:
    # coI_q(X:Y:Z)=I_q(X:Y)-I_q(X:Y|Z)
    # co-information
    co_I = I_2D(q.sum(axis=2)) - I_3D_cond_z(q)
      
    # compute the decomposition
    #
    # shared information is the coinformation of the optimal
    # distribution
    SI = co_I

    I_xz = I_2D(p.sum(axis=1))
    I_yz = I_2D(p.sum(axis=0))
    I_xy_z = I_2D(p.reshape((dimx*dimy,dimz)))

    # first output unique
    UI_X = I_xz - SI
    
    # second output unique
    UI_Y = I_yz - SI

    # output synergy
    CI = I_xy_z - UI_X - UI_Y - SI

    return SI, CI, UI_X, UI_Y
