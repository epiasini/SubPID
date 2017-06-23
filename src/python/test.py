import numpy as np
from information_decomposition_linearized import lin_pid

aa=np.int(0);

lstep = 0.05
l_values = np.arange(0.1, 1.01, lstep)
l_n = l_values.size

I_shar=np.zeros(l_n);
I_uny=np.zeros(l_n);
I_unz=np.zeros(l_n);
I_syn=np.zeros(l_n);

I_shar_z=np.zeros(l_n);
I_uny_z=np.zeros(l_n);
I_unz_z=np.zeros(l_n);
I_syn_z=np.zeros(l_n);

#Y and Z are the dice, X is the weighted sum of the two


# summed-dice dimensions
dimy=np.int(6);
dimz=np.int(6);
 
alpha=np.int(1); # relative ratio between the sum coefficients for summing the dice
 
dimx=np.int(np.ceil(dimy+dimz*alpha-1-alpha)+1);



#make X the new target
dimx_x=dimz;
dimy_x=dimy;
dimz_x=dimx;


for (k,l) in enumerate(l_values): # loop over different correlations between the dice

    
    p_yz = np.zeros((dimy,dimz))
    
    # setting correlations between the dice
    for yy in range(0,dimy):
        for zz in range(0,dimz):
            p_yz[yy,zz]=l/36+(1-l)/6*np.float(zz==yy)
    
    p_yz=p_yz/p_yz.sum() # p(y,z), joint probability of the two dice alone

    p_x_yz=np.zeros((dimx,dimy,dimz)) # p(x|y,z)

    #build p(x|y,z) analytically
    for xx in range(0,dimx):
        for yy in range(0,dimy):
            for zz in range(0,dimz):
                #if xx==np.ceil(zz+alpha*yy)-alpha:
                if xx==np.ceil(zz+alpha*yy):
                    p_x_yz[xx,yy,zz]=1
   
    p=np.zeros((dimx,dimy,dimz))

    #build p analytically
    for xx in range(0,dimx):
        for yy in range(0,dimy):
            for zz in range(0,dimz):
                p[xx,yy,zz]=p_x_yz[xx,yy,zz]*p_yz[yy,zz]

    p=p/p.sum() #final normalization
    
    
    # compute linearised PID
    [SI, CI, UI_Y, UI_Z] = lin_pid(p.T, verbose=False)
    I_syn[k] = CI
    I_shar[k] = SI
    I_uny[k] = UI_Y
    I_unz[k] = UI_Z
    

    # adapt definitions to those expected by tartu
    P = dict()
    P_z = dict()
    for x in range(0,dimx):
        for y in range(0,dimy):
            for z in range(0,dimz):
                P[(x,y,z)] = p[x,y,z]
                P_z[(z,y,x)] = p[x,y,z]

    print('\nLambda: {:.2f}'.format(l))
    #%time [_, _, _, I_syn[k],I_shar[k],I_uny[k],I_unz[k]] = synergy_tartu.solve_PDF(pdf=P,verbose=False,feas_eps=1e-15,feas_eps_2=1e-15)
    #%time [_, _, _, I_syn_z[aa],I_shar_z[aa],I_uny_z[aa],I_unz_z[aa]] = synergy_tartu.solve_PDF(pdf=P_z)
    
