function out = subatomic_pid(p)
    % SUBATOMIC_PID(p) subatomic partial information decomposition of a trivariate probability distribution.
    % 
    % Calculates the atomic and subatomic information-theoretic measures of
    % all three PID lattices of a system (X,Y,Z), as defined in Pica et al,
    % 2017, "Invariant components of synergy, redundancy, and unique
    % information among three variables"
    % (https://arxiv.org/abs/1706.08921).
    %
    % p must be a three-dimensional array representing the joint
    % probability distribution p(x,y,z) of the three discrete variables X,
    % Y and Z. The values assumed by X, Y and Z are implicitly mapped to
    % indices such that, for instance, p(1,3,2) is the probability of
    % X=1,Y=3,Z=2, and so on.
    %
    % All measures are expressed in bit. Output values are given as fields
    % of a structure. Field names include the minimum number of references
    % to X, Y and Z needed to be unambiguous, but no more than that. So,
    % for instance, here is how some of the output variables' names
    % translate in term of the notation used by Pica et al in the paper:
    % 
    % SI_x = SI(X:{Y,Z})
    %
    % CI_x = CI(X:{Y,Z})
    %
    % UI_x_y = UI(X:{Y\Z})
    %
    % RSI_x = RSI(Y<X>Z) = RSI(Z<X>Y)
    %
    % RCI_x = RCI(Y<X>Z) = RCI(Z<X>Y)
    %
    % RUI_x = RUI(Y<X>Z) = RUI(Z<X>Y)
    %
    % IRSI_x_y_z = IRSI(X<Y<Z)
    %
    % SR_x = SR(X:{Y,Z})
    %
    % NSR_x = NSR(X:{Y,Z})
    
    out = struct();
    
    [out.SI_x, out.CI_x, out.UI_x_z, out.UI_x_y] = pid(permute(p,[3 2 1]));
    [out.SI_y, out.CI_y, out.UI_y_x, out.UI_y_z] = pid(permute(p,[1 3 2]));
    [out.SI_z, out.CI_z, out.UI_z_x, out.UI_z_y] = pid(p);
        
    out.RSI_x = min(out.SI_z, out.SI_y);
    out.RSI_y = min(out.SI_z, out.SI_x);
    out.RSI_z = min(out.SI_x, out.SI_y);
    
    out.RCI_x = min(out.CI_z, out.CI_y);
    out.RCI_y = min(out.CI_z, out.CI_x);
    out.RCI_z = min(out.CI_x, out.CI_y);
    
    out.RUI_x = min(out.UI_y_z, out.UI_z_y);
    out.RUI_y = min(out.UI_x_z, out.UI_z_x);
    out.RUI_z = min(out.UI_x_y, out.UI_y_x);
    
    out.IRSI_x_z_y = out.SI_x - out.RSI_z;
    out.IRSI_x_y_z = out.SI_x - out.RSI_y;
    out.IRSI_y_z_x = out.SI_y - out.RSI_z;
    out.IRSI_y_x_z = out.SI_y - out.RSI_x;
    out.IRSI_z_x_y = out.SI_z - out.RSI_x;
    out.IRSI_z_y_x = out.SI_z - out.RSI_y;
    
    out.SR_x = max(out.RSI_y, out.RSI_z);
    out.SR_y = max(out.RSI_x, out.RSI_z);
    out.SR_z = max(out.RSI_x, out.RSI_y);
    
    out.NSR_x = out.SI_x - out.SR_x;
    out.NSR_y = out.SI_y - out.SR_y;
    out.NSR_z = out.SI_z - out.SR_z;

end