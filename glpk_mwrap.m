function coeff = glpk_mwrap(c, A, b, ctype, sense, param)
    
    [n_constraints, n_vars] = size(A);
    ctype_map = {'<=', '=', '>='};
    ctype_values = NaN(1, n_constraints);
    ctype_values(ctype=='U') = 1; % <=
    ctype_values(ctype=='S') = 2; % =
    ctype_values(ctype=='L') = 3; % >=
    
    
    
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
    
    %% bound description (not implemented!)
    header_bounds = {'Bounds'};
    bound_components = {};
    for v=1:n_vars
        bound_components{v} = sprintf('x_%d free', v);
    end
    
    bounds = strjoin([header_bounds, bound_components], '\n ');
    
    %% assemble problem description
    fid = fopen('problem.test.lp','wt');
    fprintf(fid, '%s\n%s\n%s', objective, constraints, bounds);
    fclose(fid);
    
    %% call standalone glpk
    command = sprintf('glpsol --lp problem.test.lp -w solution.test');
    system(command);
    
    %% read back results
    
    coeff = 0;
end