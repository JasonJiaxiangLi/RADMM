function [x, Dx,y, z,sigma, ALM_iter, ssn_to_iter,primfeas,update_flag] = PSSNAL_dl(primfeas,m, ID, c, D,DTD, PD,Q_C, x_init, Dx,y,z, e, lambda, maxItr, ALM_tol,ppa_tol,print_inner,sigma_0,update_flag)

x = x_init;
%ccT = c*c';
QQT =  Q_C*Q_C';
%primfeas = 1;
Q_Cx = Q_C'*x;
ssn_to_iter = 0;
ls = 0;
sigma = sigma_0;
for ALM_iter = 1:maxItr
    % SSN tolerance parameter
    delta_k = 0.99^ALM_iter;
    sub_tol = 0.99^ALM_iter;
    %linesearch parameter
    delta = 0.5;
    mu = 0.1;
    %% update x
    
    [x, Dx, prox_w,Q_Cx,iter_inner, ls_inner,rank,primfeas,normgrad] = PSSNCG_dl(c, D, e ,m,  ID, DTD, QQT, Q_C,Q_Cx, PD, lambda, sigma, x, Dx, y, z , delta, mu, delta_k,sub_tol,ppa_tol);
    ssn_to_iter = ssn_to_iter + iter_inner;
    ls = ls+ls_inner;
    
    %% update y
   % cx_e = xc -e;
    y = y + sigma*(Q_Cx-e);
    %% update z
    Dx_u = Dx - prox_w;
    z = z + sigma*(Dx_u);
    pd_gap = normgrad;

    if print_inner == 1
        fprintf('ALM_it:%2d,SSN_it:%2d,avg_SSN_ls:%2.1f, pr_fea:%1.2e  pd_gap:%1.2e norm_gd:%1.2e subtol:%1.2e  sigma:%1.2e rank:%5d  \n', ...
            ALM_iter, iter_inner, ls_inner/(1e-20+iter_inner),  primfeas, pd_gap,normgrad, max(delta_k*primfeas,ppa_tol),sigma,rank);
    end
    
    update_flag = update_flag + 1;
    if update_flag == 4
        sigma = min(1e6,sigma*3);
        update_flag = 0;
    end
    
    if iter_inner >= 10      % sigma too large
        sigma = max(1,sigma/3);
        update_flag = 0;
    end
    
   % if (max(primfeas, pd_gap)<= ALM_tol && (norm(x)>= 1-1e-10) && (norm(Dx,1)- F_p <= max(primfeas, pd_gap))) || (max(primfeas, pd_gap)<= ppa_tol)%-1e-7*norm(x-c)^2)
    if (max(primfeas, pd_gap)<= ALM_tol && (norm(x)>= 1-1e-10) ) || (max(primfeas, pd_gap)<= ppa_tol)
        break;
    end
    
    
end

end