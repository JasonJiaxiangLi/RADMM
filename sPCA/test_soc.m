clc; close all; clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min -0.5*trace(X.'*H*X) + rho / 2 * \|X - X0\|_{fro}^2 + mu \|X\|_1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test on proximal gradient method
n = 300; p = 50;
K = n;
A = randn(n, K); A = orth(A);
S = diag(abs(randn(K,1)));
H = A*S*A.'; % H: PSD matrix

rho = 1e3; eta = 1e-3; mu = 1;
X0 = randn(n, p);
X0 = orth(X0); 
F = @(X) -0.5*trace(X.'*H*X) + mu*sum(sum(abs(X))) + rho / 2 * norm(X - X0, 'fro')^2;
grad = @(X) -H*X + rho*(X - X0);

disp("prox grad")
X = zeros(n, p);
for i=1:50
    grad_f = grad(X);
    grad_map = (X - wthresh(X - eta*grad_f, 's', mu * eta)) / eta;
    disp(i+ "-th step: "+ F(X) + ", norm of gradient mapping: " + norm(grad_map, 'fro'));
%     if norm(grad_map, 'fro') < 1e-8
%         break;
%     end
    X = X - eta * grad_map;
end

disp("subgrad")
X = zeros(n, p); eta = 1e-4;
for i=1:50
    subg = -H*X + mu * sign(X) + rho * (X - X0);
    disp(i+ "-th X step for SOC: "+ F(X) + ", norm: " + norm(subg, 'fro'));
%     if norm(subg, 'fro') < 1e-8
%         break;
%     end
    X = X - eta * subg;
end
