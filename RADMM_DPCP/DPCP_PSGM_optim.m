function [t,B,angle_PSGM,time_PSGM,fval_PSGM] = DPCP_PSGM_optim(Xtilde,c,mu_min,maxiter,S,max_time)

% solves min_B ||Xtilde^T B||_1 s.t. B^T B=I
% INPUT
% Xtilde  : DxN data matrix of N data points of dimension D
% c        : Dimension of the orthogonal complement of the subspace.
%            To fit a hyperplane use c=1.
% mu_min  : minimum step size. Typicall is set to 10^(-15)
% maxiter : Maximal number of iterations. Typically set to 200.

% Parameters:
% alpha and beta: parameters for linear search. Typically is set to
%                  alpha = 0.001 and beta = 1/2.
% mu_0    : Initialization. Typically is set to 10^(-2).
%%%%
% OUTPUT
% t        : time used
% distances: Distance of each point to the estimated subspace.
% B        : Dxc matrix containing in its columns an orthonormal basis for
%            the orthogonal complement of the subspace.


% mu_min = 1e-15;
% maxiter = 200;
total_iter = 0;
%mu_0 = 1e-2; 
mu_0 = 1e-2; 

alpha = 1e-3;
beta = 1/2;
%beta = 0.6;
[D, N] = size(Xtilde);
obj = @(b)norm(Xtilde'*b,1);

% initialization
% [B_0,~] = eigs(Xtilde*Xtilde',c,'SM');

[B_0, diag_0] = eig(Xtilde*Xtilde');
[~, ind] = sort(diag(diag_0));
B_0 = B_0(:, ind(1:c));

%bo = normc(randn(D,1));
obj_total = 0;
count = 1;
% angle_PSGM(count) = abs(asin(norm(B_0.'*S)));
% fval_PSGM(count) = obj(B_0);
% time_PSGM(count) = 0;
% count = count + 1;
B = B_0;

time = tic;
for j = 1:c % separately solve each column
    i = 1;
    b = B(:,j);
    mu = mu_0;
    if j == 1
        obj_old = obj(b);
        while mu>mu_min && i<= maxiter
            i = i+1;
            grad = Xtilde*( sign(Xtilde'*b));
            
            %grad = sum( sign(b'*Xtilde).*Xtilde, 2);
            grad_norm = norm(grad)^2;
            %%% line search
            bk = b - mu*grad;
            while (obj( bk ./ norm(bk) )> obj_old - alpha*mu*grad_norm)&& mu>mu_min
                mu = mu*beta;
                bk = b - mu*grad;
            end
            b = bk ./ norm(bk);
            
            B(:, j) = b;
            obj_old = obj(b);
            
        end
    else
%         b = normc(b - B*(B'*b)); 
        [V, ~] = qr(B);
        V = V(:, j:end);
        Y = V'*Xtilde;
%         q = V'*b;
        q = randn(D-j+1,1); q = q/norm(q);
        obj = @(q)norm(Y'*q,1);
        obj_old = obj(q);
        while mu>mu_min && i<= maxiter
            i = i+1;            
            %%% line search
            %grad = sum( sign(b'*Xtilde).*Xtilde, 2);
            grad = Y*( sign(Y'*q));        
%             grad = grad - B*(B'*q);
            grad_norm = norm(grad)^2;
            qk = q - mu*grad;
            while ( obj( qk/norm(qk))> obj_old - alpha*mu*grad_norm)&& mu>mu_min
                mu = mu*beta;
                qk = q - mu*grad;
            end
            q = qk / norm(qk); 
            obj_old = obj(q);
                
            if toc(time) > max_time
                break;
            end
            
            if j == c
                B(:,j) = V*q;
    
                angle_PSGM(count) = abs(asin(norm(B'*S)));
                fval_PSGM(count) = norm(Xtilde'*B,1);
                time_PSGM(count) = toc(time);
                count = count + 1;
            end
        end
        b = V*q;
        
        
    end
    obj_total = obj_total + obj_old;
    total_iter = total_iter + i;
    B(:,j) = b;
    
end
mean_iter = total_iter/c;
t = toc(time);
fprintf('-------PSGM  stops after : %3d steps, principal angle: %e, fval: %e\n', total_iter, abs(asin(norm(B'*S))), norm(Xtilde'*B,1));
% distances = zeros(1,N);
% for j = 1 : N
%     distances(j) = norm(B' * Xtilde(:,j));
% end

end