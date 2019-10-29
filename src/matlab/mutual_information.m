function [mi_sr, mi_rc] = mutual_information(p_src)

%--------------------------------------------------------------------------
% Mutual Information between S and R
% MI(S:R) = H(S) + H(R) - H(S,R)
%--------------------------------------------------------------------------
% Marginal probability distributions
ps  = squeeze(sum(sum(p_src,2),3));
pr  = squeeze(sum(sum(p_src,1),3));
psr = squeeze(sum(p_src,3));
% H(S)
hs  = -dot(nonzeros(ps), log2(nonzeros(ps) + eps));
% H(R)
hr  = -dot(nonzeros(pr), log2(nonzeros(pr) + eps));
% H(S,R)
hsr = -dot(nonzeros(psr), log2(nonzeros(psr) + eps));
% Mutual Information between S and R
mi_sr  = hs + hr - hsr;

%--------------------------------------------------------------------------
% Mutual Information between R and C
% MI(R:C) = H(R) + H(C) - H(R,C)
%--------------------------------------------------------------------------
% Marginal probability distributions
pc  = squeeze(sum(sum(p_src,1),2));
prc = squeeze(sum(p_src,1));
% H(C)
hc  = -dot(nonzeros(pc), log2(nonzeros(pc) + eps));
% H(R,C)
hrc = -dot(nonzeros(prc), log2(nonzeros(prc) + eps));
% Mutual Information between R and C
mi_rc  = hr + hc - hrc;
