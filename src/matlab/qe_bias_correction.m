function [h0]=qe_bias_correction(S,R)
 
    eps=10^-(17);
    
    ntr = numel(R);
    R_values=unique(R);
    S_values=unique(S);
    
    N_bins=length(R_values);
    N_s=length(S_values);
    

    idx=randperm(ntr);
    ntr2=floor(ntr/2);
    ntr4=floor(ntr/4);
    r21=idx(1:ntr2);
    r22=idx(ntr2+1:2*ntr2);
    
    if length(R)>1
        R21=R(r21);
        R22=R(r22);
    else
        R21=ones(1,ntr2);
        R22=ones(1,ntr2);
    end
    
    
    if length(S)>1
        S21=S(r21);
        S22=S(r22);
    else
        S21=ones(1,ntr2);
        S22=ones(1,ntr2);
    end
    
    
    r41=idx(1:ntr4);
    r42=idx(ntr4+1:2*ntr4);
    r43=idx(2*ntr4+1:3*ntr4);
    r44=idx(3*ntr4+1:4*ntr4);
    
    
    if length(R)>1
        R41=R(r41);
        R42=R(r42);
        R43=R(r43);
        R44=R(r44);
    else
        R41=ones(1,ntr4);
        R42=ones(1,ntr4);
        R43=ones(1,ntr4);
        R44=ones(1,ntr4);
    end
    
    if length(S)>1
        S41=S(r41);
        S42=S(r42);
        S43=S(r43);
        S44=S(r44);
    else
        S41=ones(1,ntr4);
        S42=ones(1,ntr4);
        S43=ones(1,ntr4);
        S44=ones(1,ntr4);
    end
    
    for ss=1:N_s
        for rr=1:N_bins
            prs(ss,rr)=sum( (R==R_values(rr)).* (S==S_values(ss)) );
        end
    end
    prs=prs/sum(prs(:));
    
    for ss=1:N_s
        for rr=1:N_bins
            p21(ss,rr)=sum( (R21==R_values(rr)).* (S21==S_values(ss)) );
        end
    end
    p21=p21/sum(p21(:));
    
    for ss=1:N_s
        for rr=1:N_bins
            p22(ss,rr)=sum( (R22==R_values(rr)).* (S22==S_values(ss)) );
        end
    end
    p22=p22/sum(p22(:));
    
    
    for ss=1:N_s
        for rr=1:N_bins
            p41(ss,rr)=sum( (R41==R_values(rr)).* (S41==S_values(ss)) );
        end
    end
    p41=p41/sum(p41(:));
    
    for ss=1:N_s
        for rr=1:N_bins
            p42(ss,rr)=sum( (R42==R_values(rr)).* (S42==S_values(ss)) );
        end
    end
    p42=p42/sum(p42(:));
    
    for ss=1:N_s
        for rr=1:N_bins
            p43(ss,rr)=sum( (R43==R_values(rr)).* (S43==S_values(ss)) );
        end
    end
    p43=p43/sum(p43(:));
    
    for ss=1:N_s
        for rr=1:N_bins
            p44(ss,rr)=sum( (R44==R_values(rr)).* (S44==S_values(ss)) );
        end
    end
    p44=p44/sum(p44(:));
    
    
    %            p21=probrs(spk,r21,t,M);
    %            p22=probrs(spk,r22,t,M);
    %            p41=probrs(spk,r41,t,M);
    %            p42=probrs(spk,r42,t,M);
    %            p43=probrs(spk,r43,t,M);
    %            p44=probrs(spk,r44,t,M);
    
    hdt=0;
    for ss=1:N_s
        for rr=1:N_bins
            hdt=hdt-prs(ss,rr).*log2(prs(ss,rr)+eps);
        end
    end
    
    %            hdt=-sum(prs.*log2(prs+eps));
    
    
    
    
    %h21=-sum(p21.*log2(p21+eps));
    
    h21=0;
    for ss=1:N_s
        for rr=1:N_bins
            h21=h21-p21(ss,rr).*log2(p21(ss,rr)+eps);
        end
    end
    
    
    
    h22=0;
    for ss=1:N_s
        for rr=1:N_bins
            h22=h22-p22(ss,rr).*log2(p22(ss,rr)+eps);
        end
    end
    
    h41=0;
    for ss=1:N_s
        for rr=1:N_bins
            h41=h41-p41(ss,rr).*log2(p41(ss,rr)+eps);
        end
    end
    
    h42=0;
    for ss=1:N_s
        for rr=1:N_bins
            h42=h42-p42(ss,rr).*log2(p42(ss,rr)+eps);
        end
    end
    
    h43=0;
    for ss=1:N_s
        for rr=1:N_bins
            h43=h43-p43(ss,rr).*log2(p43(ss,rr)+eps);
        end
    end
    
    h44=0;
    for ss=1:N_s
        for rr=1:N_bins
            h44=h44-p44(ss,rr).*log2(p44(ss,rr)+eps);
        end
    end
    
    h4=(h41+h42+h43+h44)/4;
    h2=(h21+h22)/2;
    
    h0=lagrange_vec([1/ntr4 1/ntr2 1/ntr],[h4 h2 hdt]);
    
    
end
