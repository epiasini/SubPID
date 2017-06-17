function [I_shar,I_syn,I_unx,I_uny,q_opt]=PID_code(p,accuracy,method,lin_accuracy)

%inputs: discrete 3-variate probability distribution p, its dimensions dimx, dimy, dimz, an upper bound on the desired accuracy of the outputs (in bit)
%method: - 'cvx' if you want to use cvx - more reliable numerical result but much slower
         %- 'linprog' if you want to use the matlab function linprog - less reliable numerical result (in high dimensions) but faster
         % - 'glpk' the best yet
%outputs: the 4 PID atoms of I_shar (redundancy SI(Z:{X;Y})), I_unx (UI(Z:X\Y)), I_uny (UI(Z:Y\X)), I_syn (synergy CI(Z:{X;Y})). Thus, Z is the target and X, Y are the sources.

%close all
[dimx, dimy, dimz] = size(p);
        
if dimx==1 
    
    p23 = reshape(p(1,:,:),dimy,dimz);
    I_yz = p23 .* log2(p23 ./ repmat(sum(p23), [dimy 1]) ./ repmat(sum(p23,2), [1 dimz]));
    I_yz = sum(I_yz(p23 > 0));
    
    I_shar=0;
    I_syn=0;
    I_unx=0;
    I_uny=I_yz;
    q_opt=1;
    
elseif dimy==1
    
    p13 = reshape(p(:,1,:),dimx,dimz);
    I_xz = p13 .* log2(p13 ./ repmat(sum(p13), [dimx 1]) ./ repmat(sum(p13,2), [1 dimz]));
    I_xz = sum(I_xz(p13 > 0));
    
    I_shar=0;
    I_syn=0;
    I_unx=I_xz;
    I_uny=0;
    q_opt=1;
    
else

GAMMA=zeros(dimx,dimy,dimz,dimx-1,dimy-1,dimz); % this is the concatenation of all the (dimx-1)*(dimy-1)*(dimz) Gamma matrices defined in Bertschinger2013, each of which has
                                                  %the same dimensions (dimx, dimy, dimz) as the input p.

for zz=1:dimz
    for xx=1:dimx-1    
        for yy=1:dimy-1
        GAMMA(xx,yy,zz,xx,yy,zz) = 1;
        GAMMA(xx+1,yy,zz,xx,yy,zz) = -1;
        GAMMA(xx,yy+1,zz,xx,yy,zz) = -1;
        GAMMA(xx+1,yy+1,zz,xx,yy,zz) = 1;
        end
    end
end


check=0;% when check==1, the algorithm is over and the output is returned

parameters=zeros(1,ceil(dimz*(dimx-1)*(dimy-1))); % the coefficients of the matrix q in the basis of the Gamma matrices

iter=0; % counts the iterations of the algorithm

q=p; % the starting point of the algorithm is trivially set to the input p. I have tried smarter starting points but it seems the algorithm does not improve.

coeff_prev=parameters';% when iter=0, the coefficients of the iteration -1 are trivially set equal to zero too.
        
%co_I=[];% vector of the coinformation coI_q(X;Y;Z)

%coeff_tot_prev=zeros(ceil(dimz*(dimx-1)*(dimy-1)),1);% to use as a starting point for next linear optimization
                    
while check==0 % iteration loop
 
      q(q<0)=0;% eliminate tiny negative entries in q which result from the limited numerical precision
        
        %I_q(X:Y)
    q12 = sum(q, 3);
    I_xy = q12 .* log2(q12 ./ repmat(sum(q12), [dimx 1]) ./ repmat(sum(q12,2), [1 dimy]));
    I_xy = sum(I_xy(q12 > 0));
        
%       I_q(X:Y|Z)
    I_cond_xy_z = q .* log2(q ./ repmat(sum(sum(q), 2), [dimx dimy 1]) ./ ...
                   ( repmat(sum(q,2), [1 dimy 1]) ./ repmat(sum(sum(q), 2), [dimx dimy 1]) .* ...
                     repmat(sum(q),   [dimx 1 1]) ./ repmat(sum(sum(q), 2), [dimx dimy 1]) ) );
    I_cond_xy_z = sum(I_cond_xy_z(q > 0));

     %co_I=[co_I;I_xy-I_cond_xy_z]; % update coI_q
     co_I=  I_xy-I_cond_xy_z;%to make it faster
	
     if iter==0
         co_I_in=co_I;
     end	

    %Franke-Wolf optimization algorithm
    %1) determine search direction
   
    %calculating the gradient: we have an analytical expression of the gradient of the object function

    %deriv=zeros(dimx-1,dimy-1,dimz);

%     for zz=1:dimz
%        for xx=1:dimx-1
%            for yy=1:dimy-1
%                deriv(xx,yy,zz)=log2(q(xx,yy,zz)*q(xx+1,yy+1,zz))-log2(q(xx,yy+1,zz)*q(xx+1,yy,zz))+log2(sum(q(xx,yy+1,:)).*sum(q(xx+1,yy,:)))-log2(sum(q(xx,yy,:)).*sum(q(xx+1,yy+1,:)));
%            end
%        end
%     end

    deriv = log2(q(1:dimx-1,1:dimy-1,:) .* q(2:dimx,2:dimy,:)) - ...
            log2(q(1:dimx-1,2:dimy,:) .* q(2:dimx,1:dimy-1,:)) + ...
            log2( repmat(sum(q(1:dimx-1,2:dimy,:), 3), [1 1 dimz]) .* repmat(sum(q(2:dimx,1:dimy-1,:), 3), [1 1 dimz]) ) - ...
            log2( repmat(sum(q(1:dimx-1,1:dimy-1,:), 3), [1 1 dimz]) .* repmat(sum(q(2:dimx,2:dimy,:), 3), [1 1 dimz]) );
    
    %get rid of nonsense values of deriv coming from finite numerical precision
    deriv(isnan(deriv))=0;
    deriv(isinf(deriv))=0;

    %stores the coefficients of the q of the current iteration. For each value of z there is a coeff, then coeff_tot concatenates all coeffs.
    coeff_tot=zeros(ceil((dimx-1)*(dimy-1)),dimz);
    
    %the optimization can be formally divided into dimz optimizations, one for each value of the variable Z. This loop could thus be parallelized
    %par
    for zz=1:dimz

        %the constraints on q, for each z, are implemented via the inequality A*coeff<=b
        b=zeros(ceil(2*dimx*dimy),1);
		A=zeros(ceil(2*dimx*dimy),ceil((dimx-1)*(dimy-1)));

        %the number of constraints to be imposed
        count=1;

        for xx=1:dimx-1
            for yy=1:dimy-1

                    % set the constraints for the top-left entries q(1,1,zz)
                if (xx==1 && yy==1)

                    A(count,(floor(xx)-1)*(dimy-1)+yy)=-1;
                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy)=1;
				
                    b(count)=p(xx,yy,zz);
                    b(count+ceil(dimx*dimy))=1-p(xx,yy,zz);

                    count=count+1;
                end
               
                % bottom right entries
                if (xx==dimx-1 && yy==dimy-1)

                    A(count,(floor(+xx)-1)*(dimy-1)+yy)=-1;
                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy)=1;
				
                    b(count)=p(xx+1,yy+1,zz);
                    b(count+ceil(dimx*dimy))=1-p(xx+1,yy+1,zz);

                    count=count+1;
                end

                 % top right entries
                if (xx==1 && yy==dimy-1) 

                    A(count,(floor(+xx)-1)*(dimy-1)+yy)=1;
                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy)=-1;
				
                    b(count)=p(xx,yy+1,zz);
                    b(count+ceil(dimx*dimy))=1-p(xx,yy+1,zz);

                    count=count+1;
                end

                %bottom left entries
                
                if (xx==dimx-1 && yy==1)

                    A(count,(floor(+xx)-1)*(dimy-1)+yy)=1;
                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy)=-1;
				
                    b(count)=p(xx+1,yy,zz);
                    b(count+ceil(dimx*dimy))=1-p(xx+1,yy,zz);

                    count=count+1;
                end

                 % top row entries, from left to right
                if xx==1 && dimy>2 && yy<dimy-1

                    A(count,(floor(+xx)-1)*(dimy-1)+yy)=1;
					A(count,(floor(+xx)-1)*(dimy-1)+yy+1)=-1;
					
					A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy)=-1;
					A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy+1)=1;
                    
                    
                    b(count)=p(xx,yy+1,zz);
                    b(count+ceil(dimx*dimy))=1-p(xx,yy+1,zz);
                    count=count+1;

                end

                 % bottom row entries, from right to left
                if xx==dimx-1 && dimy>2 && yy<dimy-1

                    A(count,(floor(+xx)-1)*(dimy-1)+yy)=-1;
                    A(count,(floor(+xx)-1)*(dimy-1)+yy+1)=1;
			    			    
                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy)=1;
                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy+1)=-1;
								
                    b(count)=p(xx+1,yy+1,zz);
                    b(count+ceil(dimx*dimy))=1-p(xx+1,yy+1,zz);
                    count=count+1;
                end

                % left-most column, top-down
                if yy==1 && dimx>2 && xx<dimx-1

                    A(count,(floor(+xx)-1)*(dimy-1)+yy)=1;
                    A(count,(floor(+xx+1)-1)*(dimy-1)+yy)=-1;

                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy)=-1;
                    A(count+ceil(dimx*dimy),(floor(+xx+1)-1)*(dimy-1)+yy)=1;
								
                    b(count)=p(xx+1,yy,zz);
                    b(count+ceil(dimx*dimy))=1-p(xx+1,yy,zz);
				    count=count+1;

                end
%
%                % right-most column, top-down
                if yy==dimy-1 && dimx>2 && xx<dimx-1

                    A(count,(floor(+xx)-1)*(dimy-1)+yy)=-1;
                    A(count,(floor(+xx+1)-1)*(dimy-1)+yy)=1;
								
                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy)=1;
                    A(count+ceil(dimx*dimy),(floor(+xx+1)-1)*(dimy-1)+yy)=-1;

                    b(count)=p(xx+1,yy+1,zz);
                    b(count+ceil(dimx*dimy))=1-p(xx+1,yy+1,zz);
                    count=count+1;
                end
%                 

%                 % internal
                if dimx>2 && dimy>2 && xx>1 && yy>1 && xx<=dimx-1 && yy<=dimy-1

                    A(count,(floor(+xx)-1)*(dimy-1)+yy)=-1;
                    A(count,(floor(+xx-1)-1)*(dimy-1)+yy)=1;
                    A(count,(floor(+xx)-1)*(dimy-1)+yy-1)=1;
                    A(count,(floor(+xx-1)-1)*(dimy-1)+yy-1)=-1;

                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy)=1;
                    A(count+ceil(dimx*dimy),(floor(+xx-1)-1)*(dimy-1)+yy)=-1;
                    A(count+ceil(dimx*dimy),(floor(+xx)-1)*(dimy-1)+yy-1)=-1;
                    A(count+ceil(dimx*dimy),(floor(+xx-1)-1)*(dimy-1)+yy-1)=1;

                    b(count)=p(xx,yy,zz);
                    b(count+ceil(dimx*dimy))=1-p(xx,yy,zz);
                    count=count+1;

                end

            end
        end
        
        %    problem = repmat( struct('f',deriv_zz, 'Aineq',A, 'bineq',b, 'aeq',[],'beq',[],'ub',[],'lb',[],'x0',0,'solver','linprog','options',optimoptions('linprog','Display','off','OptimalityTolerance',lin_accuracy,'MaxIterations',10^5)), 1, dimz);
    

        
        %extract the portion of deriv pertaining to zz and adapt deriv_zz to the multiplication deriv_zz*coeff
                deriv_zz=permute(deriv(:,:,zz),[2 1 3]);
                deriv_zz=deriv_zz(:)';
                

                if isequal(method,'cvx')

                    cvx_begin quiet
                    cvx_precision(lin_accuracy) %low

                    variable coeff(ceil((dimx-1)*(dimy-1)))
                    minimize( deriv_zz*coeff )
                    subject to
                    A*coeff <= b
                    cvx_end

                    elseif isequal(method,'linprog')    
                        
                    %problem = struct();%use with parfor
                        
              
                    
                    problem.f=deriv_zz;
                    problem.Aineq=A;
                    problem.bineq=b;
                    problem.aeq=[];
                    problem.beq=[];
                    problem.ub=[];
                    problem.lb=[];
                    
                    problem.x0=0;%coeff_tot_prev(ceil((zz-1)*(dimx-1)*(dimy-1))+1:ceil((zz-1)*(dimx-1)*(dimy-1))+dimx-1+dimy-1-1,1);
                    problem.solver='linprog';
                    problem.options = optimoptions('linprog','Display','off','OptimalityTolerance',lin_accuracy);
                    %MaxIterations should be left as it is, otherwise the
                    %algorithms are not accurate
                    %The default method is 'interior-point'
                    %'Algorithm','dual-simplex') gives wrong results with naive use
                    % 'sqp'
                    % 'active-set'

                    coeff=linprog(problem);
                    
                    elseif isequal(method,'glpk') 
                                                                        
                        %param.dual=1;
                        
                      % param = sdpsettings('verbose',0,'solver','GLPK'); % GLPK is selected as the solver
                       param.lpsolver=2;%Interior point method. The simplex method called with param.lpsolver=1 or 3 doesn't work as well. 
                       param.msglev=0;
                       param.itcnt=10^10;
                       param.mpsobj=0;
                       param.mpsinfo=0;
                       
                       %param.scale=3; % doesn't change things
                       %param.dual=1; % doesn't change things
                       %param.presol=0;% doesn't change things
                       %param.branch=3;% doesn't change things
                       %param.pprocess=0;% doesn't change things
                       %param.usecuts=5;% doesn't change things
                       %param.binarize=1;% doesn't change things
                       
                       warning off;
                       
                       coeff = glpk(deriv_zz, A, b, [], [], repmat('U',1,length(b)), repmat('C',1,length(deriv_zz)), 1,param);
                    
                    elseif isequal(method,'lpsolve')
                        
                    coeff = lp_solve(deriv_zz,A,b,repmat(-1,1,length(b)),repmat(-1,1,length(deriv_zz)));%,vlb,vub,xint,scalemode,keep)    
                    
                end

                coeff_tot(:,zz)=coeff;% update coeff_tot with the coeff from the current zz
                
    end
           
    coeff_tot=coeff_tot(:);
    
    %if linprog doesn't find the final solution with the desired accuracy,
    %quit the algorithm and set the output q_opt=0.
    if size(coeff_tot,1)<ceil(dimz*(dimx-1)*(dimy-1))
        coeff_tot=coeff_prev;
        q=0;
        keyboard
    end
    
    p_k=p;

    for ind_gamma=1:ceil(dimz*(dimx-1)*(dimy-1))
           
        xx=mod(ceil(ind_gamma/(dimy-1))-1,dimx-1)+1;
        yy=mod(ind_gamma-1,dimy-1)+1;
        zz=ceil(ind_gamma/((dimx-1)*(dimy-1)));

        p_k=p_k+coeff_tot(ind_gamma,1).*GAMMA(:,:,:,xx,yy,zz);

    end
      
       deriv=permute(deriv,[2 1 3]);
       deriv=deriv(:)';
        
    %set the stopping criterion based on the duality gap, see Stratos;
    %iter must be larger than 1 because sometimes deriv takes 2 iters to
    %get different than zero, which is always its initial value.
    if iter>1 && (dot(deriv,coeff_prev-coeff_tot)<=accuracy)

        check=1; %exit the algorithm
        q_opt=q; % output the optimal distribution
    else
        
     %fixed increment      
        gamma_k=2/(iter+2);% this is the simplest version of Franke-Wolf algorithm, I have implemented several others below and I want to test them better
    
        q=q+gamma_k*(p_k-q); % update the q for next iteration

        
        coeff_prev=coeff_prev+gamma_k*(coeff_tot-coeff_prev); % update coeff_prev for next iteration
        
        iter=iter+1;
    
    end

end
    
I_shar=co_I; % get the output redundancy from the last coI_q

while I_shar<co_I_in-accuracy
   [I_shar,I_syn,I_unx,I_uny,q_opt]=PID_code_line_search(p,accuracy,method,lin_accuracy);
end

I_xz=0;
for i=1:dimx%x
    for j=1:dimz%z

        if sum(p(i,:,j))>0
            I_xz=I_xz+sum(p(i,:,j))*log2(sum(p(i,:,j))/(sum(sum(p(:,:,j)))*sum(sum(p(i,:,:)))));
        end

    end
end

I_unx=I_xz-I_shar;% first output unique

I_yz=0;
for k=1:dimy%y
    for j=1:dimz%z

        if sum(p(:,k,j))>0
            I_yz=I_yz+sum(p(:,k,j))*log2(sum(p(:,k,j))/(sum(sum(p(:,:,j)))*sum(sum(p(:,k,:)))));
        end

    end
end


I_uny=I_yz-I_shar;%second output unique

I_xy_z=0;
for i=1:dimx%x
    for k=1:dimy%y
        for j=1:dimz%z

            if p(i,k,j)>0
                I_xy_z=I_xy_z+p(i,k,j)*log2(p(i,k,j)/(sum(p(i,k,:))*sum(sum(p(:,:,j)))));
            end

        end
    end
end

I_syn=I_xy_z-I_xz-I_yz+I_shar;%output synergy

end

end
