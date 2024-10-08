%% RADMM with Moreau envelope
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementation of the Moreau-envelope Riemannian ADMM
% Test on the DPCP problem
% Problem: DPCP
% Manifold: Stiefel manifold St(n, p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear

%% Problem Generating
D = 70; n = D;
c = 5; p = c;
d = D - c;
trial_number = 5;
early_stopping = 1; % if 1: use f(x-f(x_pre)) < 10^-8
N = 500; % inlier number
M = 1167;  % outlier number

max_iter = 6000; max_iter_soc = 3000;
F_val_soc_avg = zeros([1,max_iter_soc]);
F_val_madmm_avg = zeros([1,max_iter]);
F_val_radmm_avg = zeros([1,max_iter]);
cpu_time_soc = zeros([trial_number,max_iter_soc]); cpu_time_soc(1) = eps;
cpu_time_madmm = zeros([trial_number,max_iter]); cpu_time_madmm(1) = eps;
cpu_time_radmm = zeros([trial_number, max_iter]); cpu_time_radmm(1) = eps;
vio_soc_avg = zeros([1,max_iter_soc]);
vio_madmm_avg = zeros([1,max_iter]);
vio_radmm_avg = zeros([1,max_iter]);

min_val = 0;

f_val_soc_avg = 0;
f_val_madmm_avg = 0;
f_val_radmm_avg = 0;

for k = 1:trial_number
    rng('shuffle');
    S = orth(randn(D,d));
%     X = normc( S*randn(d,N) );
%     O = normc(randn(D,M));
    X_ground = S*randn(d,N);
    O = randn(D,M);
    Xtilde = [X_ground O];
    fprintf('Inlier number: %d, outlier number: %d, tiral: %d\n', N, M, k);
    Y = normc(Xtilde);
    F = @(X) sum(sum(abs(Y'*X)));

    iter0 = max_iter; iter1 = max_iter; iter2 = max_iter;
    
    % random initialize
    X0 = orth(randn(n, p));
    fprintf('Fval at the initial: %f\n', F(X0));
    F_val_soc(1) = F(X0) ;
    F_val_madmm(1) = F(X0) ;
    F_val_radmm(1) = F(X0) ;
    
    %% SOC
    X = X0; W = X0;
    Lambda = zeros(size(X));
    eta = 5e-6; rho = 1e3;

    for iter=2:max_iter_soc
        temp_F = @(X) F(X) + rho / 2 * norm(X - W + Lambda, "fro")^2;
        admm_start = tic;

        % X step, subgradient step
        for i=1:100
            subg = Y*sign(Y.'*X) + rho * (X - W + Lambda);
            % disp(i+ "-th X step for SOC, norm: " + norm(subg, 'fro') + ", X fval: " + temp_F(X));
            if norm(subg, 'fro') < 1e-8
                break;
            end
            X = X - eta * subg;
        end

        % W step: a projection step
        [U,~,V] = svd(X + Lambda);
        W = U*eye(n,p)*V.';

        % Lambda step
        Lambda = Lambda + (X - W);

        elapsed_time = toc(admm_start);

        % Value update
        F_val_soc(iter) = F(W) ;
        vio_soc_avg(iter) = vio_soc_avg(iter) + norm(W - X, 'fro');
        if early_stopping && iter > 2 && abs(F_val_soc(iter-1) - F_val_soc(iter)) <= 1e-6
            break
        end

        cpu_time_soc(k,iter) = cpu_time_soc(k,iter) + elapsed_time;
        cpu_time_soc(k,iter+1) = cpu_time_soc(k,iter);

        % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
    end
    f_val_soc_avg = f_val_soc_avg + F_val_soc(iter);
    iter0 = min(iter, iter0);
    
    %% MADMM
    X = X0; W = Y.'*X0;
    Lambda = zeros(size(W));
    eta = 1e-6; rho = 5e1;

    for iter=2:max_iter
        admm_start = tic;

        % X step: a Riemannian gradient step
        for i=1:100
            gx = rho*Y*(Y.'*X - W + Lambda);
            rgx = proj_stiefel(gx, X);
            if norm(rgx, 'fro') < 1e-8
                break;
            end
            X = retr_stiefel(-eta*rgx, X);
            %fprintf('inner iter: %d, X step subgrad:%f\n', i, norm(rgx));
        end

        % W step: a l1 minimization step
        W = wthresh(Y.'*X + Lambda ,'s', 1/rho);

        % Lambda step
        Lambda = Lambda + (Y.'*X - W);

        elapsed_time = toc(admm_start);

        % Value update
        F_val_madmm(iter) = F(X) ;
        vio_madmm_avg(iter) = vio_madmm_avg(iter) + norm(Y.'*X - W, 'fro');
        if early_stopping && iter > 2 && abs(F_val_madmm(iter-1) - F_val_madmm(iter)) <= 1e-6
            break
        end

        cpu_time_madmm(k,iter) = cpu_time_madmm(k,iter) + elapsed_time;
        cpu_time_madmm(k,iter+1) = cpu_time_madmm(k,iter);

        % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
    end
    f_val_madmm_avg = f_val_madmm_avg + F_val_madmm(iter);
    iter1 = min(iter, iter1);
    
    %% RADMM
    X = X0; W = Y.'*X0; Z = W;
    Lambda = zeros(size(W));
    eta = 1e-4; rho = 5e1; gamma = 1e-10;
    
    for iter=2:max_iter
        admm_start = tic;

        % X step: a Riemannian gradient step
        for i=1:1
            gx = Y*Lambda + rho*Y*(Y.'*X - Z);
            rgx = proj_stiefel(gx, X);
            if norm(rgx, 'fro') < 1e-8
                break;
            end
            X = retr_stiefel(-eta*rgx, X);
            %fprintf('inner iter: %d, X step subgrad:%f\n', i, norm(rgx));
        end

        % Z step (also update W)
        W = wthresh(Y.' * X + Lambda/rho,'s', (1 + rho * gamma)/rho);
        Z = (W/gamma + Lambda + rho * Y.' * X) / (1/gamma + rho);

        % Lambda step
        Lambda = Lambda + rho * (Y.'*X - Z);

        elapsed_time = toc(admm_start);

        % Value update
        F_val_radmm(iter) = F(X) ;
        vio_radmm_avg(iter) = vio_radmm_avg(iter) + norm(Y.'*X - W, 'fro');
        if early_stopping && iter > 2 && abs(F_val_radmm(iter-1) - F_val_radmm(iter)) <= 1e-6
            break
        end

        cpu_time_radmm(k,iter) = cpu_time_radmm(k,iter) + elapsed_time;
        cpu_time_radmm(k,iter+1) = cpu_time_radmm(k,iter);

        % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
    end
    f_val_radmm_avg = f_val_radmm_avg + F_val_radmm(iter);
    iter2 = min(iter, iter2);
    
    min_among_all = min([min(F_val_soc(1:iter0)), min(F_val_madmm(1:iter1)), min(F_val_radmm(1:iter2))]);
    min_val = min_val + min_among_all;
    % l = size(F_val_soc);
    % for i=1:l(2)
    %     F_val_soc(i) = F_val_soc(i) - min_among_all;
    % end
    % l = size(F_val_madmm);
    % for i=1:l(2)
    %     F_val_madmm(i) = F_val_madmm(i) - min_among_all;
    % end
    % l = size(F_val_radmm);
    % for i=1:l(2)
    %     F_val_radmm(i) = F_val_radmm(i) - min_among_all;
    % end
    F_val_soc_avg(1:iter0) = F_val_soc_avg(1:iter0) + F_val_soc(1:iter0);
    F_val_madmm_avg(1:iter1) = F_val_madmm_avg(1:iter1) + F_val_madmm(1:iter1);
    F_val_radmm_avg(1:iter2) = F_val_radmm_avg(1:iter2) + F_val_radmm(1:iter2);
    
    % f_val_soc_avg = f_val_soc_avg + min_among_all + F_val_soc(iter0);
    % f_val_madmm_avg = f_val_madmm_avg + min_among_all + F_val_madmm(iter1);
    % f_val_radmm_avg = f_val_radmm_avg + min_among_all + F_val_radmm(iter2);
end

avg = trial_number;

min_val = min_val / avg;

F_val_soc_avg = (F_val_soc_avg/avg);
F_val_madmm_avg = (F_val_madmm_avg/avg);
F_val_radmm_avg = (F_val_radmm_avg/avg);

vio_soc_avg = (vio_soc_avg/avg);
vio_madmm_avg = (vio_madmm_avg/avg);
vio_radmm_avg = (vio_radmm_avg/avg);

cpu_time_soc = sum(cpu_time_soc,1)/avg;
cpu_time_madmm = sum(cpu_time_madmm,1)/avg;
cpu_time_radmm = sum(cpu_time_radmm,1)/avg;

disp("error of SOC, MADMM and RADMM: ")

disp([vio_soc_avg(iter0 - 1), vio_madmm_avg(iter1 - 1), vio_radmm_avg(iter2 - 1)])

disp("CPU time for SOC, MADMM and RADMM: ")

disp([cpu_time_soc(iter0 - 1), cpu_time_madmm(iter1 - 1), cpu_time_radmm(iter2 - 1)]);

disp("function value for output SOC, MADMM and RADMM: ")

disp([f_val_soc_avg/avg, f_val_madmm_avg/avg, f_val_radmm_avg/avg]);

%% Plots
figure0 = figure(1);
clf
semilogy(F_val_soc_avg(1:iter0) - min_val, '-.','LineWidth',2); hold on;
semilogy(F_val_madmm_avg(1:iter1) - min_val, '-.','LineWidth',2); hold on;
semilogy(F_val_radmm_avg(1:iter2) - min_val,'LineWidth',2); hold on;
xlabel('Iteration','interpreter','latex','FontSize',18); ylabel('$f(x)-f^*$','interpreter','latex','FontSize',18);
legend('SOC', 'MADMM', 'RADMM');
legend('Location','best','FontSize',20);
filename = "dpcp_soc_madmm_n_" + n + "_p_" + p + "_fval_" + early_stopping + ".pdf";
saveas(figure0, filename);
% figure0.show()

figure1 = figure(2);
clf
loglog(cpu_time_soc(1:iter0), F_val_soc_avg(1:iter0) - min_val, '-.','LineWidth',2); hold on;
loglog(cpu_time_madmm(1:iter1), F_val_madmm_avg(1:iter1) - min_val, '-.','LineWidth',2); hold on;
loglog(cpu_time_radmm(1:iter2), F_val_radmm_avg(1:iter2) - min_val,'LineWidth',2); hold on;
xlabel('CPU time','interpreter','latex','FontSize',18); ylabel('$f(x)-f^*$','interpreter','latex','FontSize',18);
legend('SOC', 'MADMM', 'RADMM');
legend('Location','best','FontSize',20);
filename = "dpcp_soc_madmm_cpu_time_n_" + n + "_p_" + p + "_fval_" + early_stopping + ".pdf";
saveas(figure1, filename);
% figure1.show()

figure2 = figure(3);
clf
semilogx(cpu_time_soc(1:iter0), vio_soc_avg(1:iter0), '-.','LineWidth',2); hold on;
semilogx(cpu_time_madmm(1:iter1), vio_madmm_avg(1:iter1), '-.','LineWidth',2); hold on;
semilogx(cpu_time_radmm(1:iter2), vio_radmm_avg(1:iter2),'LineWidth',2); hold on;
xlabel('CPU time','interpreter','latex','FontSize',18); ylabel('Constrain violation','interpreter','latex','FontSize',18);
legend('SOC', 'MADMM', 'RADMM');
legend('Location','best','FontSize',20);
% saveas(figure1, filename);
% figure1.show()

figure3 = figure(4);
clf
semilogy(F_val_soc_avg(1:iter0), '-.','LineWidth',2); hold on;
semilogy(F_val_madmm_avg(1:iter1), '-.','LineWidth',2); hold on;
semilogy(F_val_radmm_avg(1:iter2),'LineWidth',2); hold on;
xlabel('Iteration','interpreter','latex','FontSize',18); ylabel('$f(x)$','interpreter','latex','FontSize',18);
legend('SOC', 'MADMM', 'RADMM');
legend('Location','best','FontSize',20);
filename = "dpcp_soc_madmm_n_" + n + "_p_" + p + "_fval_only_" + early_stopping + ".pdf";
saveas(figure3, filename);
% figure0.show()

figure4 = figure(5);
clf
loglog(cpu_time_soc(1:iter0), F_val_soc_avg(1:iter0), '-.','LineWidth',2); hold on;
loglog(cpu_time_madmm(1:iter1), F_val_madmm_avg(1:iter1), '-.','LineWidth',2); hold on;
loglog(cpu_time_radmm(1:iter2), F_val_radmm_avg(1:iter2),'LineWidth',2); hold on;
xlabel('CPU time','interpreter','latex','FontSize',18); ylabel('$f(x)$','interpreter','latex','FontSize',18);
legend('SOC', 'MADMM', 'RADMM');
legend('Location','best','FontSize',20);
filename = "dpcp_soc_madmm_cpu_time_n_" + n + "_p_" + p + "_fval_only_" + early_stopping + ".pdf";
saveas(figure4, filename);
% figure1.show()