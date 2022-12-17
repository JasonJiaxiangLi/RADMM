function [X_radmm, angle_radmm, time_radmm, fval_admm] = radmm_DPCP(B,option)
%%% min ||B*X||_1 s.t X'*X=I_c
% parameters


[m,d] = size(B); % size of B
radmm_iter = 0;
maxiter = option.maxiter; % number of maximum iteration
max_inner_iter = option.max_inner_iter;
rho = option.rho; 
gamma = option.gamma; 
eta = option.eta;
h=@(X) norm(B*X, 1); % objective function
Bt = B';  BTB = B'*B;

% create the initial
if ~isfield(option,'phi_init')
    [phi_init, diag_0] = eig(BTB);
    [~, ind] = sort(diag(diag_0));
    option.phi_init = phi_init(:, ind(1:option.c));
end
%%
%option.phi_init = randn(d,option.c);
X_radmm = option.phi_init; % initial point

% initialization
X = X_radmm ;
Lambda = zeros(m, option.c); Z = B*X; 
count = 1;
angle_radmm(count) = abs(asin(norm(X'*option.S)));
fval_admm(count) = h(X);
obj_old = fval_admm(count);
time_radmm(count) = 0;
count = count + 1;
time = tic;

%% main loop
for radmm_iter = 2:maxiter
    % X step: a gradient step
    for i=1:max_inner_iter
        gx = B.'*Lambda + rho*B.'*(B*X - Z);
        rgx = proj_stiefel(gx, X);
        X = retr_stiefel(-eta*rgx, X);
        % fprintf('inner iter: %d, X step subgrad:%f\n', i, norm(rgx));
        if norm(rgx,'fro')<=1e-6
            break
        end
    end

    % Z step (also update Y)
    Y = wthresh(B * X + Lambda/rho,'s', (1 + rho * gamma)/rho);
    Z = (Y/gamma + Lambda + rho * B * X) / (1/gamma + rho);

    % Lambda step
    Lambda = Lambda + rho*(B*X - Z);

    % update gamma
    %     if norm(Z-Y,'fro')>= 0.9*dist_ZY(iter-1) && gamma >= eps
    %         gamma = gamma/2;
    %     end

    %     if norm(X-Z,'fro')>= 0.9*dist_XZ(iter-1) && rho <=1e6
    %         rho = rho*2;
    %     end

    % Value update
    angle_radmm(count) = abs(asin(norm(X'*option.S)));
    fval_admm(count) = h(X);
    time_radmm(count) = toc(time);
    obj_val_radmm(radmm_iter) = h(X);
    
    % fprintf('ADMM iter: %d, XT*b - z: %f, fval: %f\n', iter, norm(Xtilde.'*b - z), obj_val_radmm(iter));
    
    if (abs(h(X) - obj_old) <= option.tol && norm(B*X - Z) <= option.tol) || toc(time) > option.max_time
        break
    end
    obj_old = fval_admm(count);
    count = count + 1;

end

X_radmm = X;
fprintf('-------RADMM iter: %3d, function value: %3d, error: %d\n',...
    radmm_iter, h(X), norm(B*X - Z));
%time_manppa = toc;

% figure
% plot(obj_val_radmm);

end

