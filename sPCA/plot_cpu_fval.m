%% RADMM with Moreau envelope: test on running time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementation of the Moreau-envelope Riemannian ADMM
% Test on the sPCA problem
% Problem: sPCA, min -1/2*tr(X^THX)+\mu*\|X\|_1=f(X)+h(X)
% Manifold: Stiefel manifold St(n, p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear

addpath misc
addpath SSN_subproblem
%% Problem Generating
n = 1500; p = 200; K = 300;
A = randn(n, K); A = A - mean(A, 1); A = A./sqrt(diag(A.'*A)).';
% S = diag(abs(randn(K,1)));
H = A*A.';
mu = 0.5;
f = @(X) -0.5*trace(X.'*H*X);
g = @(Y) mu*sum(sum(abs(Y)));
g_gamma = @(Z,gamma) mu*(g(wthresh(Z,'s',gamma))+1/(2*gamma)*norm(wthresh(Z,'s',gamma) - Z,'fro')^2);
F = @(X) f(X) + g(X);
sub_F = @(X) - H * X + mu * sign(X);

%% Algorithm
% initialization in Stiefel manifold
X = randn(n, p);
X = orth(X);
Y = X; Z = X; U0 = X;
Lambda = zeros(size(X));
eta = 1e-2; gamma = 1e-8; N = 3000; rho = 50;
grad_gamma_F = @(X) - H * X + mu/gamma * ( X - wthresh(X,'s',gamma) );

% Parameters for ManPG subproblem
t = 1/(2*max(abs(svd(A)))^2);
inner_iter = 100;
prox_fun = @(b,l,r) proximal_l1(b,l,r); % proximal function used in solving the subproblem
t_min = 1e-4; % minimum stepsize
num_linesearch = 0;
num_inexact = 0;
inner_flag = 0;
Dn = sparse(DuplicationM(p)); % vectorization for SSN
pDn = (Dn'*Dn)\Dn'; % for SSN
nu = 0.8; % penalty coefficient?
alpha = 1; % stepsize fpr ManPG
tol = 1e-8;

Lag = @(X,Y,Lambda,gamma,rho) f(X) + mu*g(Y) + trace(Lambda.'*(X-Y)) + rho/2*norm(X-Y)^2;
L_M = @(X,Z,Lambda,gamma,rho) f(X) + mu*g_gamma(Z,gamma) + trace(Lambda.'*(X-Z)) + rho/2*norm(X-Z)^2;


iter = 1;
F_val_admm = F(X); F_val_manpg = F(X); F_val_grad = F(X);
time_admm = 0; time_manpg = 0; time_grad = 0;
L_val_admm(iter) = Lag(X,Y,Lambda,gamma,rho);
MEL_val(iter) = L_M(X,Z,Lambda,gamma,rho);
% dist_XY(iter) = norm(X-Y,'fro'); dist_XZ(iter) = norm(X-Z,'fro'); dist_ZY(iter) = norm(Z-Y,'fro');
norm_admm = norm(proj(X, grad_gamma_F(X)),'fro'); norm_grad = norm_admm; norm_manpg = norm_admm;

%% ManPG
U = U0;
tic
for iter=2:N
    neg_pg = -H*U;
    if alpha < t_min || num_inexact > 10
        inner_tol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
    else
        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
    end

    % The subproblem
    if iter == 2
         [ PU,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,U,t,U + t*neg_pg,nu*t,inner_tol,prox_fun,inner_iter,zeros(p),Dn,pDn);
        %      [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    else
         [ PU,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,U,t,U + t*neg_pg,nu*t,inner_tol,prox_fun,inner_iter,Lam_x,Dn,pDn);
        %     [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    end

    if in_flag == 1   % subprolem not exact.
        inner_flag = 1 + inner_flag;
    end

    V = PU-U; % The V solved from SSN

    % projection onto the Stiefel manifold
    [T, SIGMA, S] = svd(PU'*PU);   SIGMA =diag(SIGMA);    U_temp = PU*(T*diag(sqrt(1./SIGMA))*S');

    f_trial = f(U_temp);
    F_trial = f_trial + g(U_temp);   normV=norm(V,'fro');

    %%% Without linesearch
    PU = U+alpha*V;
    % projection onto the Stiefel manifold
    [T, SIGMA, S] = svd(PU'*PU);   SIGMA =diag(SIGMA);   U_temp = PU*(T*diag(sqrt(1./SIGMA))*S');
    U = U_temp; % update

    time_manpg(iter) = toc;
    F_val_manpg(iter) = F(U);
    norm_manpg(iter) = normV;

    % if norm_manpg(iter) < tol 
    %     break;
    % end

    if abs(F_val_manpg(iter-1) - F(U)) <= 1e-8
        break
    end


end
sol_manpg = U;

%% MERADMM
tic
for iter=2:N
    % X step: a gradient step
    for i=1:1
        gx = -H*X + Lambda + rho*(X - Z);
        rgx = proj(X, gx);
        X = retr(X, -(eta)*rgx);
        % fprintf('inner iter: %d, X step subgrad:%f\n', i, norm(rgx));
    end

    % Z step (also update Y)
    Y = wthresh(X+Lambda/rho,'s',mu*(1+rho*gamma)/rho);
    Z = (Y/gamma + Lambda + rho*X) / (1/gamma + rho);

    % Lambda step
    Lambda = Lambda + rho*(X - Z);

    % update gamma
    %     if norm(Z-Y,'fro')>= 0.9*dist_ZY(iter-1) && gamma >= eps
    %         gamma = gamma/2;
    %     end

    %     if norm(X-Z,'fro')>= 0.9*dist_XZ(iter-1) && rho <=1e6
    %         rho = rho*2;
    %     end

    % Value update
    time_admm(iter) = toc;
    F_val_admm(iter) = F(X);
    L_val_admm(iter) = Lag(X,Y,Lambda,gamma,rho);
    % MEL_val(iter) = L_M(X,Z,Lambda,gamma,rho);
    % dist_XY(iter) = norm(X-Y,'fro'); dist_XZ(iter) = norm(X-Z,'fro'); dist_ZY(iter) = norm(Z-Y,'fro');
    norm_admm(iter) = norm(proj(X, grad_gamma_F(X)),'fro'); 
    % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));

    if abs(F_val_admm(iter-1) - F(X)) <= 1e-8
        break
    end

end
sol_admm = X;

%% Riemannian grad method
U = U0; eta = 1e-2;
tic
for iter=2:N
    rgrad = proj(U, grad_gamma_F(U)); % calculate the Riemannian gradient
    U = retr(U, -eta*rgrad); % update

    time_grad(iter) = toc;
    F_val_grad(iter) = F(U);
    norm_grad(iter) = norm(rgrad,'fro');

    if abs(F_val_grad(iter-1) - F(U)) <= 1e-8
        break
    end
end
sol_grad = U;

%% Plots
figure;
semilogy(time_admm(2:end),norm_admm(2:end)); hold on;
semilogy(time_manpg(2:end),norm_manpg(2:end),'-.'); hold on;
semilogy(time_grad(2:end),norm_grad(2:end),'--'); hold on;
legend('rADMM', 'ManPG', 'RGrad');
title('Norm of the convergence criterion versus CPU time');

% comparison between MERADMM and ManPG
figure;
plot(time_admm(2:end),F_val_admm(2:end)); hold on;
plot(time_manpg(2:end),F_val_manpg(2:end),'-.'); hold on;
plot(time_grad(2:end),F_val_grad(2:end),'--');
title('function values versus CPU time');
legend('RADMM', 'ManPG', 'RGrad');