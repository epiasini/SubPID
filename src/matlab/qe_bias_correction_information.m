function I = qe_bias_correction_information(X,Y, iters) 
    I = qe_bias_correction(X,1, iters) + qe_bias_correction(1,Y, iters) - qe_bias_correction(X,Y, iters);
end