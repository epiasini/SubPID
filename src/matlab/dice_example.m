function [I_shar_z, RSI_x_y_z, IRSI_x_y_z] = dice_example()
    % DICE_EXAMPLE() test for PID based on an example given in Harder 2013.
    %
    % This is for internal testing only. Variables Y and Z represent the
    % dice, X is the weighted sum of the two.
    
    tic
    
    % dice dimensions
    dimy=6;
    dimz=6;
    
    % relative ratio between the sum coefficients for summing the dice
    alpha=1;
    
    % step size by which to increase parameter controlling correlations between dice
    lambda_step=.05;
    
    % initialise data structures for storing results
    I_shar=zeros(1,ceil(1/lambda_step)+1);
    I_unx=zeros(1,ceil(1/lambda_step)+1);
    I_uny=zeros(1,ceil(1/lambda_step)+1);
    I_syn=zeros(1,ceil(1/lambda_step)+1);
    I_shar_z=zeros(1,ceil(1/lambda_step)+1);
    I_unx_z=zeros(1,ceil(1/lambda_step)+1);
    I_uny_z=zeros(1,ceil(1/lambda_step)+1);
    I_syn_z=zeros(1,ceil(1/lambda_step)+1);
    RSI_x_y_z=zeros(1,ceil(1/lambda_step)+1);
    IRSI_x_y_z=zeros(1,ceil(1/lambda_step)+1);
    
    % X is the target
    dimx=ceil(dimy+dimz*alpha-1-alpha)+1;
    
    for lambda_idx=1:ceil(1/lambda_step)+1 % loop over different levels of correlation between the dice
        
        lambda = (lambda_idx-1)*lambda_step;
        
        p_yz = zeros(dimy,dimz);
        
        % setting correlations between the dice
        for yy=1:dimy
            for zz=1:dimz
                p_yz(yy,zz)=lambda/36+(1-lambda)/6*(yy==zz);
            end
        end
        
        p_yz=p_yz/sum(sum(p_yz)); % p(y,z), joint probability of the two dice alone
        
        p_x_yz=zeros(dimx,dimy,dimz); % p(x|y,z)
        
        %build p(x|y,z) analytically
        for xx=1:dimx
            for yy=1:dimy
                for zz=1:dimz
                    
                    if xx==ceil(alpha*zz+yy)-alpha
                        p_x_yz(xx,yy,zz)=1;
                    end
                    
                end
            end
        end
        
        p=zeros(dimx,dimy,dimz);
        
        %build p analytically
        for xx=1:dimx
            for yy=1:dimy
                for zz=1:dimz
                    
                    p(xx,yy,zz)=p_x_yz(xx,yy,zz).*p_yz(yy,zz);
                    
                end
            end
        end
        
        p=p/sum(sum(sum(p)));%final normalization
        
        
        % compute PID with Z as the target variable
        [I_shar_z(lambda_idx),I_syn_z(lambda_idx),I_unx_z(lambda_idx),I_uny_z(lambda_idx)] = pid(p);
        
        % compute PID with X as the target variable
        [I_shar(lambda_idx),I_syn(lambda_idx),I_unx(lambda_idx),I_uny(lambda_idx)] = pid(permute(p,[3 2 1]));
        
        toc
        
        % compute RSI(X<-Y->Z)
        RSI_x_y_z(lambda_idx) = min(I_shar(lambda_idx),I_shar_z(lambda_idx));
        
        % compute IRSI(X<-Y--Z)
        IRSI_x_y_z(lambda_idx) = I_shar(lambda_idx)-RSI_x_y_z(lambda_idx);
        
    end
    
    toc
    
end


