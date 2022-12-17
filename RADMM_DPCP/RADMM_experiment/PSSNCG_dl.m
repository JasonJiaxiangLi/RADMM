function  [x, Dx,prox_w,Q_Cx, j, ls,rank,primfeas,normgrad] = PSSNCG_dl( c, D, e, col_m , ID, DTD, QQT,Q_C,Q_Cx, PD, lambda, sigma, x,Dx, dual_y, z , delta, mu, delta_k,sub_tol,out_tol)
% L = 0.5*||x-b||^2+||w||_1+<z,Dx-w>+y*(c'*x -1)+0.5*sigma*(||Dx-w||^2+(c'x-1)^2)
%[p,n] = size(D);
u_t = sigma*Dx + z;
prox_v  = sign(u_t).* min(lambda, abs(u_t));
prox_w = Dx + z/sigma - prox_v/sigma;
DT_prox_v = D'*prox_v;
xc =  c'*x;
%Q_Cx = Q_C'*x;
grad = x-c + DT_prox_v + sigma*(Q_C*(Q_Cx + dual_y/sigma-e)) ;
phi_org = 0.5*norm(x)^2 - xc  + lambda*norm(prox_w,1)+ 1/(2*sigma)*norm(prox_v)^2 + 0.5*sigma*norm( Q_Cx+dual_y/sigma-e)^2;
normgrad = norm(grad);
j = 0;
ls = 0;
Iter = 20;
Iter_m = 10;
rank = 0;
% primfeas = sqrt(norm(Dx - prox_w)^2 + ( xc -1)^2);
primfeas = sqrt(norm(Dx - prox_w)^2 + norm( Q_Cx -e)^2);
%ssn_tol = max(min(delta_k*primfeas,sub_tol),0.9*out_tol);
ssn_tol = max(min(delta_k*primfeas,sub_tol),out_tol);
if primfeas > 1
    ssn_tol = 1;
end
while (j<= Iter && normgrad >ssn_tol )
    %% semi_smooth_newton
    idx = (abs(sigma*Dx + z)> lambda);
    l = sum(idx);
    
    if l <= col_m/2
        
        D_idx = PD(:, idx); % choose columns in D' which is faster than rows
        V = ID + sigma*(QQT + DTD - D_idx*D_idx');
        rank = l;
        
    else
        idx_rev = (abs(sigma*Dx + z)<=lambda);
        D_idx = PD(:,idx_rev); % choose columns in D' which is faster than rows
        V  = ID + sigma*(QQT + D_idx*D_idx');
        rank = col_m -l;
    end
    
    %  newton direction
    %     [L,U] = lu(V);
    %     d = U\(L\(-grad));
    L = chol(V,'lower');
    d = L'\(L\(-grad));
    %%%% line search
    %d = -grad;
    
    g_d = grad'*d;
    
    Dd = D*d;
    dc = d'*c;
    Q_Cd = Q_C'*d;
    j = j+1;
    %     phi_org = 0.5*u'*u+b'*u+sigma/2*norm(w)^2+1/(2*L*sigma)*norm(prox_z)^2-sigma/(2*L)*norm(D*w)^2;
    
    m = 0;
    %for  mm = 1: Iter_m
    stepsize = 1;
    
    x_bar = x + stepsize*d;
    x_barc = xc + stepsize*dc;
    Dx_bar = Dx + stepsize*Dd;
    Q_Cx_bar = Q_Cx + stepsize*Q_Cd;
    %[phi_1,   prox_v] = C_phi_Py( e, lambda, sigma, x_bar, x_barc, Dx_bar, y, z);
    prox_v  = sign(sigma*Dx_bar + z).* min( lambda, abs(sigma*Dx_bar + z));
    prox_w = Dx_bar + z/sigma - prox_v/sigma;
    phi_1 = 0.5*norm(x_bar)^2 - x_barc  + lambda*norm(prox_w,1)+ 1/(2*sigma)*norm(prox_v)^2 + 0.5*sigma*norm( Q_Cx_bar+dual_y/sigma-e)^2;
    
    while (phi_1 >  phi_org + mu*stepsize*g_d)&& m < Iter_m
        stepsize = delta*stepsize;
        x_bar = x + stepsize*d;
        x_barc = xc + stepsize*dc;
        Dx_bar = Dx + stepsize*Dd;
        Q_Cx_bar = Q_Cx + stepsize*Q_Cd;
        prox_v  = sign(sigma*Dx_bar + z).* min( lambda, abs(sigma*Dx_bar + z));
        prox_w = Dx_bar + z/sigma - prox_v/sigma;
        phi_1 = 0.5*norm(x_bar)^2 - x_barc  + lambda*norm(prox_w,1)+ 1/(2*sigma)*norm(prox_v)^2 + 0.5*sigma*norm( Q_Cx_bar+dual_y/sigma-e)^2;
        m  = m+1;
    end
    
    ls = ls+m;  % number of linesearch
    x = x_bar;
    xc = x_barc;
    Dx = Dx_bar;
    Q_Cx = Q_Cx_bar;
    DT_prox_v = D'*prox_v;
    % new grad
    grad = x-c + DT_prox_v + sigma*(Q_C*(Q_Cx + dual_y/sigma-e)) ;
    
    normgrad = norm(grad);
    phi_org = phi_1 ;
    
    primfeas = sqrt(norm(Dx - prox_w)^2 + norm( Q_Cx -e)^2);
    ssn_tol = max(min(delta_k*primfeas,sub_tol),out_tol);
    
    %fprintf('SSN_it:%2d, l_sear:%2d, pr_fea:%1.2e,  norm_gd:%1.2e,  rank:%5d\n', j, m,  primfeas,normgrad,rank);
end

end