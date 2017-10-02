function I = qe_bias_correction_information(X,Y, iters)
    if nargin<3
        iters = 1;
    end
    I = qe_bias_correction(X,1, iters) + qe_bias_correction(1,Y, iters) - qe_bias_correction(X,Y, iters);
end