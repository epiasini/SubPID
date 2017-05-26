function solution_coords = glpk_mwrap(c, A, b, ctype, sense, method)
    % min/maximise c'x, subject to Ax{<=,=,>=}b. 'ctype' is a char array
    % specifying the type of constraint for each dimension of the variable
    % x ('U':<=, 'S':=, 'L':>=). 'sense' specifies if the objective
    % function shosuld be maximized (1) or minimised (-1). 'method' is the
    % optimisation method used by glpk, and can be 'simplex' or 'interior'.
    
    [n_constraints, n_vars] = size(A);
    ctype_map = {'<=', '=', '>='};
    ctype_values = NaN(1, n_constraints);
    ctype_values(ctype=='U') = 1; % <=
    ctype_values(ctype=='S') = 2; % =
    ctype_values(ctype=='L') = 3; % >=
    
    switch method
        case 'simplex'
            scan_string = 'j %d %*s %f %*f';
        case 'interior'
            scan_string = 'j %d %f %*f';
    end
    
    %% objective description string
    % (note we're goint to a new line at each variable because of the
    % limitation on max line length)
    if sense==1
        header_objective = {'Minimize', 'obj:'};
    elseif sense==-1
        header_objective = {'Maximize', 'obj:'};
    end
    obj_components = {};
    idx = 0;
    for v=1:n_vars
        if c(v) ~= 0
            idx = idx + 1;
            obj_components{idx} = sprintf('%+.15e x_%d', c(v), v);
        end
    end
    % make sure we always have something in the objective
    if idx == 0
        obj_components{1} = '0 x_1';
    end
    
    objective = strjoin([header_objective, obj_components], '\n ');
        
    %% constraint description
    header_constraints = {'Subject To'};
    cons_components = {};
    idx = 0;
    for c_idx=1:n_constraints
        if nnz(A(c_idx,:))>0
            idx = idx + 1;
            
            this_constraint_components = {};
            this_idx = 0;
            for v=1:n_vars
                if A(c_idx,v) ~= 0
                    this_idx = this_idx + 1;
                    this_constraint_components{this_idx} = sprintf('%+.15e x_%d', A(c_idx,v), v);
                end
            end
            
            cons_components{idx} = sprintf('%s\n%s %.15e',...
                strjoin(this_constraint_components, '\n'),...
                ctype_map{ctype_values(c_idx)},...
                b(c_idx));
        end
    end
    
    constraints = strjoin([header_constraints, cons_components], '\n ');
    
    %% bound description (not implemented! variables will be unbounded)
    header_bounds = {'Bounds'};
    bound_components = cell(1, n_vars);
    for v=1:n_vars
        bound_components{v} = sprintf('x_%d free', v);
    end
    
    bounds = strjoin([header_bounds, bound_components], '\n ');
    
    %% assemble problem description
    fid = fopen('problem.test.lp','wt');
    fprintf(fid, '%s\n%s\n%s', objective, constraints, bounds);
    fclose(fid);
    
    %% call standalone glpk
    command = sprintf('glpsol --%s --lp problem.test.lp -w solution.test', method);
    system(command);
    
    %% read back results
    fid = fopen('solution.test', 'r');
    tline = fgetl(fid);
    solution_coords = NaN(n_vars, 1);
    while ischar(tline)
        if tline(1)=='j'
            scanned = textscan(tline, scan_string);
            solution_coords(scanned{1}) = scanned{2};
        end
        tline = fgetl(fid);
    end
    fclose(fid);
end