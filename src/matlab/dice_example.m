function dice_example

tic

aa=1;
lambdastep=.05;
lambda=0.;

accuracy=0.01;%with glpk, setting accuracy to 0.001 doesn't improve things, which are already good enough

I_shar=zeros(1,ceil(1/lambdastep)+1);
I_unx=zeros(1,ceil(1/lambdastep)+1);
I_uny=zeros(1,ceil(1/lambdastep)+1);
I_syn=zeros(1,ceil(1/lambdastep)+1);
I_shar_z=zeros(1,ceil(1/lambdastep)+1);
I_unx_z=zeros(1,ceil(1/lambdastep)+1);
I_uny_z=zeros(1,ceil(1/lambdastep)+1);
I_syn_z=zeros(1,ceil(1/lambdastep)+1);

RSI_x_y_z=zeros(1,ceil(1/lambdastep)+1);
IRSI_x_y_z=zeros(1,ceil(1/lambdastep)+1);

%Y and Z are the dice, X is the weighted sum of the two

% summed-dice dimensions
  dimy=6;
  dimz=6;
% 
 alpha=1; % relative ratio between the sum coefficients for summing the dice
% 

% X is the target
  dimx=ceil(dimy+dimz*alpha-1-alpha)+1;
% 
%

%use if Z is the new target
dimx_z=dimz;%
dimy_z=dimy;%
dimz_z=dimx;


while aa<=ceil(1/lambdastep)+1 % loop over different correlations between the dice

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

    p=permute(p,[3 2 1]);
    
    %[I_shar(aa),I_syn(aa),I_unx(aa),I_uny(aa),q_1]=PID_code(dimz,dimy,dimx,p,accuracy,'linprog',10^-6);
    
    %[I_shar(aa),I_syn(aa),I_unx(aa),I_uny(aa),q_1]=PID_code(dimz,dimy,dimx,p,0.01,'cvx',10^-3);
    
    [I_shar(aa),I_syn(aa),I_unx(aa),I_uny(aa),q_1]=pid(p,accuracy,'glpk',0);
    
    %[I_shar(aa),I_syn(aa),I_unx(aa),I_uny(aa),q_1]=PID_code(dimz,dimy,dimx,p,accuracy,'lpsolve',0);
    
toc

    p_z=permute(p,[3 2 1]);

   %[I_shar_z(aa),I_syn_z(aa),I_unx_z(aa),I_uny_z(aa),q_3]=PID_code(dimz_z,dimy_z,dimx_z,p_z,accuracy,'linprog',10^-6);
   
   %[I_shar_z(aa),I_syn_z(aa),I_unx_z(aa),I_uny_z(aa),q_3]=PID_code(dimz_z,dimy_z,dimx_z,p_z,0.01,'cvx',10^-3);
        %or use 'cvx' as the solving method

   [I_shar_z(aa),I_syn_z(aa),I_unx_z(aa),I_uny_z(aa),q_3] = pid(p_z,accuracy,'glpk',0.);
    
    %[I_shar_z(aa),I_syn_z(aa),I_unx_z(aa),I_uny_z(aa),q_3]=PID_code(dimz_z,dimy_z,dimx_z,p_z,accuracy,'lpsolve',0.);
   
        
    RSI_x_y_z(aa)=min(I_shar(aa),I_shar_z(aa));% in the draft paper, this is RSI between X and Z passing through Y
    IRSI_x_y_z(aa)=I_shar(aa)-RSI_x_y_z(aa);% this is the IRSI between X and Z passing through Y


    aa=aa+1;

    lambda=lambda+lambdastep;

end

toc

end


