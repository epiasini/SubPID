function [I_shar,I_syn,I_unx,I_uny,q_opt]=PID_code(dimx,dimy,dimz,p,accuracy,method)

%inputs: discrete 3-variate probability distribution p, its dimensions dimx, dimy, dimz, an upper bound on the desired accuracy of the outputs (in bit)
%method: - 'cvx' if you want to use cvx - more reliable numerical result but much slower
         %- 'linprog' if you want to use the matlab function linprog - less reliable numerical result (in high dimensions) but faster
%outputs: the 4 PID atoms of I_shar (redundancy SI(Z:{X;Y})), I_unx (UI(Z:X\Y)), I_uny (UI(Z:Y\X)), I_syn (synergy CI(Z:{X;Y})). Thus, Z is the target and X, Y are the sources.

close all

tic


%basis=zeros(dimx,dimy,dimz);%set the matrix dimensions of the GAMMA matrices

GAMMA=zeros(dimx,dimy,dimz,dimx-1,dimy-1,dimz); % this is the concatenation of all the (dimx-1)*(dimy-1)*(dimz) Gamma matrices defined in Bertschinger2013, each of which has
                                                  %the same dimensions (dimx, dimy, dimz) as the input p.

for zz=1:dimz
    for xx=1:dimx-1    
        for yy=1:dimy-1
        %GAMMA(:,:,:,xx,yy,zz)=basis;
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

%build q
%for ind_gamma=1:ceil(dimz*(dimx-1)*(dimy-1))
           
%xx=mod(ceil(ind_gamma/(dimy-1))-1,(dimx-1))+1;
%yy=mod(ind_gamma-1,dimy-1)+1;
%zz=ceil(ind_gamma/((dimx-1)*(dimy-1)));
            
%q=q+parameters(1,ind_gamma).*GAMMA(:,:,:,xx,yy,zz);

%end

coeff_prev=parameters';% when iter=0, the coefficients of the iteration -1 are trivially set equal to zero too.
        
co_I=[];% vector of the coinformation coI_q(X;Y;Z)

%S_t=parameters';
 
while check==0 % iteration loop
 
      q(q<0)=0;% eliminate tiny negative entries in q which result from the limited numerical precision
        
      I_xy=0;% I_q(X:Y)
        for i=1:dimx%x
            for k=1:dimy%y
       
                if sum(q(i,k,:))>0
                    I_xy=I_xy+sum(q(i,k,:))*log2(sum(q(i,k,:))/(sum(sum(q(:,k,:)))*sum(sum(q(i,:,:)))));
                end
        
            end
        end         
        
      I_cond_xy_z=0;% I_q(X:Y|Z)
        for i=1:dimx%x
           for k=1:dimy%y
             for j=1:dimz%z

                if q(i,k,j)>0
                   I_cond_xy_z=I_cond_xy_z+q(i,k,j)*log2(q(i,k,j)/sum(sum(q(:,:,j)))/(sum(q(i,:,j))/sum(sum(q(:,:,j)))*sum(q(:,k,j))/sum(sum(q(:,:,j)))));
                end

             end
           end
        end


     co_I=[co_I;I_xy-I_cond_xy_z]; % update coI_q
        

    %Franke-Wolf optimization algorithm
    %1) determine search direction

    I_zy_x=0; % I_q(Z:Y|X), which is the function minimized by q_opt, the final q which should be found by the algorithm
        for i=1:dimx%x
           for k=1:dimy%y
             for j=1:dimz%z
                if q(i,k,j)>0
                   I_zy_x=I_zy_x+q(i,k,j)*log2(q(i,k,j)/(sum(sum(q(:,:,j)))*sum(q(i,k,:))));
                end
             end
           end
        end

%calculating the gradient: we have an analytical expression of the gradient of the object function

    deriv=zeros(dimx-1,dimy-1,dimz);

    for zz=1:dimz
        for xx=1:dimx-1
            for yy=1:dimy-1
                deriv(xx,yy,zz)=log2(q(xx,yy,zz)*q(xx+1,yy+1,zz))-log2(q(xx,yy+1,zz)*q(xx+1,yy,zz))+log2(sum(q(xx,yy+1,:)).*sum(q(xx+1,yy,:)))-log2(sum(q(xx,yy,:)).*sum(q(xx+1,yy+1,:)));
            end
        end
    end

    %get rid of nonsense values of deriv coming from finite numerical precision
    deriv(isnan(deriv))=0;
    deriv(isinf(deriv))=0;

    %stores the coefficients of the q of the current iteration. For each value of z there is a coeff, then coeff_tot concatenates all coeffs.
    coeff_tot=[];

    %the optimization can be formally divided into dimz optimizations, one for each value of the variable Z. This loop could thus be parallelized
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

                %extract the portion of deriv pertaining to zz and adapt deriv_zz to the multiplication deriv_zz*coeff
                deriv_zz=permute(deriv(:,:,zz),[2 1 3]);
                deriv_zz=deriv_zz(:)';
                

                if isequal(method,'cvx')

                    cvx_begin quiet
                    cvx_precision(0.0001) %low

                    variable coeff(ceil((dimx-1)*(dimy-1)))
                    minimize( deriv_zz*coeff )
                    subject to
                    A*coeff <= b
                    cvx_end

                    elseif isequal(method,'linprog')

                    problem.f=deriv_zz;
                    problem.Aineq=A;
                    problem.bineq=b;
                    problem.aeq=[];
                    problem.beq=[];
                    problem.ub=[];
                    problem.lb=[];
                    problem.x0=0;
                    problem.solver='linprog';
                    problem.options = optimoptions('linprog','Display','off','OptimalityTolerance',accuracy/10^5,'MaxIterations',10^20);
                    %The default method is 'interior-point'
                    %'Algorithm','dual-simplex') gives wrong results with naive use

                    coeff=linprog(problem);

                end

                coeff_tot=[coeff_tot;coeff];% update coeff_tot with the coeff from the current zz

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
    if (dot(deriv,coeff_prev-coeff_tot)<=accuracy)

        check=1; %exit the algorithm
        q_opt=q; % output the optimal distribution
    else
        
     %fixed increment      
        gamma_k=2/(iter+2);% this is the simplest version of Franke-Wolf algorithm, I have implemented several others below and I want to test them better
    
        q=q+gamma_k*(p_k-q); % update the q for next iteration

        coeff_prev=coeff_prev+gamma_k*(coeff_tot-coeff_prev);  % update coeff_prev for next iteration


% first alternative Franke-Wolf increment: line search. Pretty primitive version
    %line search for gamma_k
%     gamma=0;
%     I_cond_xy_z_prev=I_cond_xy_z;
%     
%     while gamma<1 %gamma=1 just gives you p_k again
%     
%     q_search=q+gamma*(p_k-q);    
%         
%        I_cond_xy_z=0;
% %conditional mutual info
%         for i=1:dimx%x
%            for k=1:dimy%y
%              for j=1:dimz%z
%                 if q_search(i,k,j)>0
%                    I_cond_xy_z=I_cond_xy_z+q_search(i,k,j)*log2(q_search(i,k,j)/sum(sum(q_search(:,:,j)))/(sum(q_search(i,:,j))/sum(sum(q_search(:,:,j)))*sum(q_search(:,k,j))/sum(sum(q_search(:,:,j)))));
%                 end
%              end
%            end
%         end   
%         
%        if  I_cond_xy_z>I_cond_xy_z_prev
%            gamma_k=gamma;
%            gamma=1;
%            I_cond_xy_z_prev=I_cond_xy_z;
%        end
%         
%     gamma=gamma+0.01;    
%     end    
%         
%     q=q+gamma_k*(p_k-q); %because I started with iter=0
%      
%     coeff_prev=coeff_prev+gamma_k*(coeff_tot-coeff_prev);        



% second alternative Franke-Wolf increment: away steps. Pretty primitive version
%      v_value_max=0;%    dot(deriv,parameters);
%      v_t=1;
%      for ss=2:size(S_t,2)
%          if v_value_max<dot(deriv,S_t(:,ss))
%              v_value_max=dot(deriv,S_t(:,ss));
%              v_t=ss;
%          end
%      end
% 
%      if dot(deriv,coeff_prev-coeff_tot)>=dot(deriv,S_t(:,v_t)-coeff_prev)
%          dir=coeff_tot-coeff_prev;
%          gamma_max=1;
%      else
%          dir=-S_t(:,v_t)+coeff_prev;
%          alpha_v_t=dot(coeff_prev,S_t(:,v_t));
%          gamma_max=alpha_v_t/(1-alpha_v_t);
%      end
%      
%         %line search for gamma_k
%     gamma=0;
%     I_cond_xy_z_prev=I_cond_xy_z;
%     
%     vec_dir=p;    
%         for ind_gamma=1:ceil(dimz*(dimx-1)*(dimy-1))
%            
%             xx=mod(ceil(ind_gamma/(dimy-1))-1,dimx-1)+1;
%             yy=mod(ind_gamma-1,dimy-1)+1;
%             zz=ceil(ind_gamma/((dimx-1)*(dimy-1)));
%             
%             
%             vec_dir=vec_dir+dir(ind_gamma,1).*GAMMA(:,:,:,xx,yy,zz);
% 
%         end
%     
%     
%     while gamma<=gamma_max
%        
%     q_search=q+gamma*(vec_dir);    
%         
%        I_cond_xy_z=0;
% %conditional mutual info
%         for i=1:dimx%x
%            for k=1:dimy%y
%              for j=1:dimz%z
%                 if q_search(i,k,j)>0
%                    I_cond_xy_z=I_cond_xy_z+q_search(i,k,j)*log2(q_search(i,k,j)/sum(sum(q_search(:,:,j)))/(sum(q_search(i,:,j))/sum(sum(q_search(:,:,j)))*sum(q_search(:,k,j))/sum(sum(q_search(:,:,j)))));
%                 end
%              end
%            end
%         end   
%         
%        if  I_cond_xy_z>I_cond_xy_z_prev
%            gamma_k=gamma;
%            gamma=1;
%            I_cond_xy_z_prev=I_cond_xy_z;
%        end
%        
%      if gamma_max>0   
%     gamma=gamma+gamma_max/1000;    
%      else
%          gamma=gamma+0.1;
%      end
%             
%             
%     end    
%         
%     
% 
%     q=q+gamma_k*(vec_dir); %because I started with iter=0
%      
%     coeff_prev=coeff_prev+gamma_k*(dir);        
%     
%     if gamma_k==1
%     S_t=[coeff_tot];
%     else
%     S_t=[S_t,coeff_tot];    
%     end
% 
%     if gamma_k==gamma_max
%     S_t(:,v_t)=[];
%     else
%     end
%         
        
        
        iter=iter+1;
    
    end

end
    
I_shar=max(co_I); % get the output redundancy from the max of the coI_q
    
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


toc

end