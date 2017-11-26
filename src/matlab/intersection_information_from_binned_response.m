function I_II=intersection_information_from_binned_response(S,R,C, R_bins)

% intersection_information_from_binned_response: Computes intersection information II of a perceptual discrimination
% dataset where the experimenter recorded, in each of n_trial trials, the stimulus S, some neural feature R, and the choice C.
%
% The information-theoretic intersection information II is defined and described in Pica et al (2017), Advances in Neural
% Information Processing, "Quantifying how much sensory information in a neural code is relevant for behavior".


% Inputs:
% S must be a one dimensional array of 1 X n_trials elements representing 
% the discrete value of the stimulus presented in each trial.

% R must be a one dimensional array of 1 X n_trials elements representing 
% the discrete value of the neural feature R recorded in each trial. The 
% recorded response feature can be multidimensional, but its dimensionality 
% should be reduced to construct the vector R. To reliably estimate II from 
% the empirical probability distribution p(s,r,c), the dimension of the 
% neural feature R should be roughly reduced to less than 
% n_trials/(4*n_stimuli*n_choices) values - though we are developing robust 
% non-parametric methods to assess the significance of II for any dimension
% of R. Here, we implemented arguably the simplest dimensionality reduction 
% of R - we discretize R into R_bins bins (R_bins can be set by the user, 
% or is conservatively set to 3 by default) that are equally populated.
%
% C must be a one dimensional array of 1 X n_trials elements representing the discrete value of the
% choice made by the animal in each trial.


if nargin<4
    R_bins=3;
end

n_trials = numel(S);

bin_edges=quantile(R, linspace(0,1,R_bins+1));
classes_indices=discretize(R, bin_edges);

R_discrete_values=unique(classes_indices);
S_values=unique(S);
C_values=unique(C);

N_r=numel(R_discrete_values);
N_s=numel(S_values);
N_c=numel(C_values);


% estimate probability distribution p(s,r,c) from 3D histogram of the input (S, R, C) occurrences
p_src=zeros(N_s,N_r,N_c);

for cc=1:N_c
    for ss=1:N_s
        for rr=1:N_r
            p_src(ss,rr,cc)=sum( (classes_indices==R_discrete_values(rr)).* (S==S_values(ss)).* (C==C_values(cc)));
        end
    end
end
%
p_src=p_src/sum(p_src(:));

I_II=intersection_information(p_src);

end