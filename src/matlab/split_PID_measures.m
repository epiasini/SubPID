function [Intersection_X,Intersection_Y,SR,MR]=split_PID_measures(p)

    % This code calculates the finer information-theoretic measures of the
    % PID lattices of a system (X,Y,Z). Intersection_X is the information
    % between Y and Z that passes through X; Intersection_Y is the
    % information between X and Z that passes through Y; the source
    % redundancy (SR) is the information about Z that is shared between X
    % and Y and stored in the X-Y correlations; the mechanistic redundancy
    % (MR) is information about Z that is shared between X and Y as it
    % backfires to X and Y because there is a synergistic mechanism, or
    % area, that combines X and Y to influence Z.
    
    % All measures are output in bit.
    
    SI_z_xy = pid(p);
    
    SI_x_yz = pid(permute(p,[3 2 1]));
    
    SI_y_xz = pid(permute(p,[2 1 3]));
    
    Intersection_Y=min(SI_z_xy,SI_x_yz);
    Intersection_X=min(SI_z_xy,SI_y_xz);
    
    SR=max(Intersection_X,Intersection_Y);
    MR=SI_z_xy-SR;

end
