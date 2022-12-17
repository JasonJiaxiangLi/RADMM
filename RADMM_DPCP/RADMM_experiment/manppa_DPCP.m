function [X_manppa, F_total, angle_manppa, time_manppa, fval_manppa] = manppa_DPCP(B,option)
%%% min ||B*X||_1 s.t X'*X=I_c
%%% j-th subproblem  min 0.5*||x^j-x^j_{k-1}||^2 + t||B*x^j||_1
%%% st. x^j'*x^j_{k-1} = 1 & x^j'*x^{l} = 0,  l=1,...,j-1
% parameters
time = tic;

[m,d] = size(B);
maxiter = option.maxiter;
h=@(BX) norm(BX,1);
%proj_span = option.proj_span;
%span = option.span;
t_min = 1e-6; % minimum stepsize
inner_tol_m = max(1e-9,option.tol);
print_inner = 1;
Bt = B';  BTB = B'*B;

if isfield(option,'print_inner')
    if any(strcmp(option.print_inner,{'off'}))
        print_inner = 0;
    end
end
if ~isfield(option,'phi_init')
    [phi_init, diag_0] = eig(BTB);
    [~, ind] = sort(diag(diag_0));
    option.phi_init = phi_init(:, ind(1:option.c));
end
%%
% option.phi_init = randn(d,option.c);
X_manppa = option.phi_init;
F_total = 0;
manppa_total_iter = 0;
ALM_total_c = 0;
ssn_total_c = 0;
Q_p = [];
count = 1;
% angle_manppa(count) = abs(asin(norm(X_manppa'*option.S)));
% fval_manppa(count) = h(B * option.phi_init);
% time_manppa(count) = 0;
count = count + 1;
time = tic;
for j = 1:option.c
    %initial point
    X = X_manppa(:,j);
    
    BX = B*X;
    F_p = h(BX);
    F(1) = F_p;
    num_linesearch = 0;
    t = option.stepsize;
    
    ssn_total = 0;
    ALM_total = 0;
    fail_flag = 0;
    
    ID = eye(d);
    sigma0 = 3000*t;
    
    Fval = 1e8;  F_b = F_p;  F_c = F_b; L = 5;
    
    %% first step
    manppa_iter = 1;
    inner_tol = max(inner_tol_m, 1e-2); % subproblem inexact;
    
    if option.exact == 1
        inner_tol = min(1e-8,inner_tol_m/2);
    end
    Q_C = [X,Q_p];
    e = zeros(j,1);  e(1) = 1;
    % subproblem
    [Y,BY, dual_y,dual_z,sigma,ALM_iter,ssn_iter,primfeas,update_flag] = PSSNAL_dl(norm(BX),m,ID, X, B, BTB,Bt,Q_C,X, BX,0,zeros(m,1),e, t, 30,...
        inner_tol,inner_tol_m,print_inner,sigma0,0) ;
    
    ALM_total = ALM_total + ALM_iter;
    ssn_total = ssn_total + ssn_iter;
    alpha = 1;
    D = Y - X; %descent direction D
    BD = BY - BX;
    Z = Y/norm(Y);
    BZ = BY/norm(Y);
    F_trial= h(BZ);
    normDsquared = norm(D,'fro')^2;
    
    %% linesearch
    while F_trial>= Fval -0.5*alpha/t*normDsquared
        alpha=0.5*alpha;
        num_linesearch = num_linesearch + 1;
        if alpha<t_min
            break;
        end
        Y = X + alpha*D;
        BY = BX + alpha*BD;
        Z = Y/norm(Y);
        BZ = BY/norm(Y);
        F_trial= h(BZ);
    end
    X = Z;  BX =BZ;
    F(manppa_iter) = F_trial;
    F_relative = abs(F_trial - F_p)/F_p;
    % angle =  abs( (span*(proj_span'*X))'* X)/norm((span*(proj_span'*X)));
    if F_relative <= option.tol && inner_tol < inner_tol_m
        return
    end
    %
    % if angle <= option.tol
    %     fprintf('-------ManPG iter: %3d, ALM iter: %2d,  subiter: %3d\n', manpg_iter, ALM_total, ssn_total);
    %     return;
    % end
    F_p  =  F_trial;
    
    %angle_ManPPA(manppa_total_iter + 1) =  asin(norm( vecnorm(X_manppa(:,1:j)'*X,2,2),'inf'));
    %% main loop
    for manppa_iter = 2:maxiter
        %t = 0.6*0.6^iter;
        %% solve subproblem
        if F_relative > 1e-3
            inner_tol = max(inner_tol_m/(manppa_iter), 1e-2); % subproblem inexact;
        else
            inner_tol = min(max(inner_tol_m/2, 0.1^manppa_iter),1e-4);
        end
        
        if option.exact == 1
            inner_tol = min(1e-8,inner_tol_m/2);
        end
        Q_C = [X,Q_p];
        e = zeros(j,1);  e(1) = 1;
        [Y,BY, dual_y,dual_z,sigma,ALM_iter, ssn_iter,primfeas,update_flag] = PSSNAL_dl(primfeas,m,ID, X, B, BTB,Bt,Q_C,X, BX, dual_y, dual_z, e,...
            t, 30, inner_tol,inner_tol_m,print_inner,sigma,update_flag) ;
        ALM_total = ALM_total + ALM_iter;
        ssn_total = ssn_total + ssn_iter;
        alpha = 1;
        D = Y - X; %descent direction D
        BD = BY - BX;
        Z = Y/norm(Y);
        BZ = BY/norm(Y);
        F_trial= h(BZ);   normDsquared = norm(D,'fro')^2;
        Y_p = Y;
        
        %%  non-monotone linesearch
        while F_trial>= Fval - 0.5*alpha/t*normDsquared %&& primfeas < 1e-7
            alpha=0.5*alpha;
            num_linesearch = num_linesearch+1;
            if alpha<t_min
                break;
            end
            Y = X + alpha*D;
            BY = BX + alpha*BD;
            Z = Y/norm(Y);
            BZ = BY/norm(Y);
            F_trial= h(BZ);
        end
        X = Z;  BX =BZ;
        %q_rot_manpg = Q' *Z;
        F(manppa_iter) = F_trial;
        F_relative = abs(F_trial - F_p)/F_p;
        F_p  =  F_trial;
        X_manppa(:,j) = X;
        if j == option.c
            angle_manppa(count) = abs(asin(norm(X_manppa'*option.S))); 
            fval_manppa(count) = h(B * X_manppa);
            time_manppa(count) = toc(time); count = count + 1;
        end
        
       % angle_ManPPA(manppa_total_iter + manppa_iter) =  asin(norm( vecnorm(X_manppa(:,1:j)'*X,2,2),'inf'));
        if F_relative <= option.tol && inner_tol < inner_tol_m
            Q_p = [Q_p,X];
            %             fprintf('-------ManPPA iter: %3d, ALM iter: %2d,  subiter: %3d\n', manppa_iter+1, ALM_total, ssn_total);
            break;
        end
        
        if toc(time) > option.max_time
            break;
        end
       
        % non-monotone
        if F_trial < F_b
            F_b= F_trial; F_c = F_b;  l=0;
        else
            F_c = max(F_c,F_trial); l=l+1;
            if l== L
                Fval = F_c; F_c =F_trial; l=0;
            end
        end
        if manppa_iter == maxiter
            % disp(['iter_sub=',  num2str(i)]);
            %             fprintf('-------ManPPA iter: %3d, ALM iter: %2d,  subiter: %3d\n', manppa_iter+1, ALM_total, ssn_total);
            fail_flag = 1;
            Q_p = [Q_p,X];
        end
        
        
    end
   
    F_total = F_total + F_trial;
    manppa_total_iter = manppa_total_iter + manppa_iter;
    ALM_total_c = ALM_total_c + ALM_total;
    ssn_total_c = ssn_total_c+ ssn_total;
    
end
manppa_mean_iter = manppa_total_iter/option.c;
ALM_mean_c = ALM_total_c/option.c;
ssn_mean_c = ssn_total_c/option.c;
fprintf('-------ManPPA iter: %3d, ALM iter: %2d,  subiter: %3d\n', manppa_total_iter, ALM_total_c, ssn_total_c);
%time_manppa = toc;
end

