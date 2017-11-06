function [I_II_corr]=qe_bias_correction_II(S,R,C,I_II_non_corrected, iters)
    
    if nargin<3
        iters = 1;
    end
    
    ntr = numel(R);
    
    R_values=unique(R);
    S_values=unique(S);
    C_values=unique(C);
    
    N_r=length(R_values);
    N_s=length(S_values);
    N_c=length(C_values);
    
    I_II_corr_iter=[];
    
for iter=1:iters
    
    idx=randperm(ntr);
    ntr2=floor(ntr/2);
    ntr4=floor(ntr/4);
    r21=idx(1:ntr2);
    r22=idx(ntr2+1:2*ntr2);
    
    
    R_21=R(r21);
    R_22=R(r22);
    S_21=S(r21);
    S_22=S(r22);
    C_21=C(r21);
    C_22=C(r22);
    
    
    r41=idx(1:ntr4);
    r42=idx(ntr4+1:2*ntr4);
    r43=idx(2*ntr4+1:3*ntr4);
    r44=idx(3*ntr4+1:4*ntr4);
    
    
    R_41=R(r41);
    R_42=R(r42);
    R_43=R(r43);
    R_44=R(r44);
    
    S_41=S(r41);
    S_42=S(r42);
    S_43=S(r43);
    S_44=S(r44);
    
    C_41=C(r41);
    C_42=C(r42);
    C_43=C(r43);
    C_44=C(r44);
    
    
    p_src_21=zeros(N_s,N_r,N_c);
    
    for cc=1:N_c
        for ss=1:N_s
            for rr=1:N_r
                p_src_21(ss,rr,cc)=sum( (R_21==R_values(rr)).* (S_21==S_values(ss)).* (C_21==C_values(cc)));
            end
        end
    end
    %
    p_src_21=p_src_21/sum(p_src_21(:));
    
    
    p_src_22=zeros(N_s,N_r,N_c);
    
    for cc=1:N_c
        for ss=1:N_s
            for rr=1:N_r
                p_src_22(ss,rr,cc)=sum( (R_22==R_values(rr)).* (S_22==S_values(ss)).* (C_22==C_values(cc)));
            end
        end
    end
    %
    p_src_22=p_src_22/sum(p_src_22(:));
    
    
    [I,II,III,IIII]=partial_info_dec(p_src_21);
    
    p_s_2=permute(p_src_21,[3 2 1]);
    %
    [I_s,II_s,III_s,IIII_s]=partial_info_dec(p_s_2);
    
    I_II_21=min(I,I_s);
    
    
    
    [I,II,III,IIII]=partial_info_dec(p_src_22);
    
    p_s_2=permute(p_src_22,[3 2 1]);
    %
    [I_s,II_s,III_s,IIII_s]=partial_info_dec(p_s_2);
    
    I_II_22=min(I,I_s);
    
    
    
    p_src_41=zeros(N_s,N_r,N_c);
    
    for cc=1:N_c
        for ss=1:N_s
            for rr=1:N_r
                p_src_41(ss,rr,cc)=sum( (R_41==R_values(rr)).* (S_41==S_values(ss)).* (C_41==C_values(cc)));
            end
        end
    end
    %
    p_src_41=p_src_41/sum(p_src_41(:));
    
    p_src_42=zeros(N_s,N_r,N_c);
    
    for cc=1:N_c
        for ss=1:N_s
            for rr=1:N_r
                p_src_42(ss,rr,cc)=sum( (R_42==R_values(rr)).* (S_42==S_values(ss)).* (C_42==C_values(cc)));
            end
        end
    end
    %
    p_src_42=p_src_42/sum(p_src_42(:));
    
    p_src_43=zeros(N_s,N_r,N_c);
    
    for cc=1:N_c
        for ss=1:N_s
            for rr=1:N_r
                p_src_43(ss,rr,cc)=sum( (R_43==R_values(rr)).* (S_43==S_values(ss)).* (C_43==C_values(cc)));
            end
        end
    end
    %
    p_src_43=p_src_43/sum(p_src_43(:));
    
    p_src_44=zeros(N_s,N_r,N_c);
    
    for cc=1:N_c
        for ss=1:N_s
            for rr=1:N_r
                p_src_44(ss,rr,cc)=sum( (R_44==R_values(rr)).* (S_44==S_values(ss)).* (C_44==C_values(cc)));
            end
        end
    end
    %
    p_src_44=p_src_44/sum(p_src_44(:));
    
    
    
    [I,II,III,IIII]=partial_info_dec(p_src_41);
    
    p_s_2=permute(p_src_41,[3 2 1]);
    %
    [I_s,II_s,III_s,IIII_s]=partial_info_dec(p_s_2);
    
    I_II_41=min(I,I_s);
    
    
    [I,II,III,IIII]=partial_info_dec(p_src_42);
    
    p_s_2=permute(p_src_42,[3 2 1]);
    %
    [I_s,II_s,III_s,IIII_s]=partial_info_dec(p_s_2);
    
    I_II_42=min(I,I_s);
    
    
    
    [I,II,III,IIII]=partial_info_dec(p_src_43);
    
    p_s_2=permute(p_src_43,[3 2 1]);
    %
    [I_s,II_s,III_s,IIII_s]=partial_info_dec(p_s_2);
    
    I_II_43=min(I,I_s);
    
    
    [I,II,III,IIII]=partial_info_dec(p_src_44);
    
    p_s_2=permute(p_src_44,[3 2 1]);
    %
    [I_s,II_s,III_s,IIII_s]=partial_info_dec(p_s_2);
    
    I_II_44=min(I,I_s);
    
    
    
    
    I_II_4=(I_II_41+I_II_42+I_II_43+I_II_44)/4;
    I_II_2=(I_II_21+I_II_22)/2;
    
    I_II_corr_iter=[I_II_corr_iter; lagrange_vec([1/ntr4 1/ntr2 1/ntr],[I_II_4 I_II_2 I_II_non_corrected])];
    
end

I_II_corr=mean(I_II_corr_iter);
    
end
