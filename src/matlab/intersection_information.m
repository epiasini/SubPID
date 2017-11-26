function [I_II, S_R_info, C_R_info, S_C_info, non_readout_sensory_info, internal_choice_info, ...
    S_C_info_from_unobserved_R]=intersection_information(p_src)

% intersection_information: Computes intersection information II of a perceptual discrimination
% dataset where the experimenter recorded, in each of n_trial trials, the stimulus S, some neural feature R, and the choice C.
% 
% The information-theoretic intersection information II is defined and described in Pica et al (2017), Advances in Neural
% Information Processing, 3687-3697, "Quantifying how much sensory information in
% a neural code is relevant for behavior".

% The definition of II also defines immediately the leftover information components:
% I(S:R)-II is sensory information that is not readout for behavior, "non_readout_sensory_info"
% I(R:C)-II is choice information that is not related to the stimulus, "internal_choice_info"
% I(S:C)-II is correspondence between stimulus and choice that is due to other neural responses 
% than the observed R, "S_C_info_from_unobserverd_R" 


% Inputs:
% p_src is a n_stimuli X n_response_values X n_choices array of the 
% empirical joint probability distribution p(s,r,c) estimated from 
% the experimental dataset. For each stimulus value s, discrete response
% value r, and choice c, p_src(s,r,c) is just the fraction of trials that 
% corresponded to the specific triplet (s,r,c).


% compute SI(C: {S;R}), UI(C: {S\R}), UI(C: {R\S})
[SI_c,~,~,UI_c_r]=partial_info_dec(p_src);

p_crs=permute(p_src,[3 2 1]);
% compute SI(S: {C;R}), UI(S:{C\R}), UI(S: {R\C})
[SI_s,~,UI_s_c,UI_s_r]=partial_info_dec(p_crs);

% compute final output intersection information
I_II=min(SI_c,SI_s);



% stimulus information available in the recorded neural response R
S_R_info=SI_s+UI_s_r;

% choice information available in the recorded neural response R
C_R_info=SI_c+UI_c_r;

% I(S:C), similar in spirit to behavioral performance
S_C_info=SI_s+UI_s_c;



% sensory information in neural response R that is not read out for behavior
non_readout_sensory_info=S_R_info-II;

% choice information in neural response R that is not related to the stimulus
internal_choice_info=R_C_info-II;

% the part of I(S:C) ("behavioral performance") that cannot be explained 
% with recorded neural feature R
S_C_info_from_unobserved_R=S_C_info-II;



end
