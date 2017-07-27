function I = qe_bias_correction_information(X,Y) 
    I = qe_bias_correction(X,1) + qe_bias_correction(1,Y) - qe_bias_correction(X,Y);
end