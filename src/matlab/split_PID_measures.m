function [RSI_x, RSI_y, SR_z, NSR_z] = split_PID_measures(p)

    % This code calculates the finer information-theoretic measures of the
    % PID lattices of a system (X,Y,Z). RSI_x is the information between Y
    % and Z that passes through X (RSI(Z<X>Y)); RSI_y is the information
    % between X and Z that passes through Y (RSI(Z<Y>X)); the source
    % redundancy SR_z is the information about Z that is shared between X
    % and Y and stored in the X-Y correlations; the non-source redundancy
    % NSR_z is information about Z that does not arise from the
    % correlations between the X and Y, and is related to the synergistic
    % information that X and Y carry about Z.
    
    % All measures are output in bit.
    
    SI_z_xy = pid(p);
    SI_x_yz = pid(permute(p,[3 2 1]));
    SI_y_xz = pid(permute(p,[2 1 3]));
    
    RSI_y = min(SI_z_xy,SI_x_yz);
    RSI_x = min(SI_z_xy,SI_y_xz);
    
    SR_z = max(RSI_x,RSI_y);
    NSR_z = SI_z_xy - SR_z;

end
