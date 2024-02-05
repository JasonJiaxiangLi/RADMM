close all; clear all;
%randn('seed',2018);rand('seed',2018)
rng('default');
%% setup
D = 70;
c = 5;
d = D - c;
trial_number = 10;

N = 500; % inlier number
M = 1167;  % outlier number

max_time = 1.0;  % maximum run in seconds
err_RADMM = 0;

% generate the problem
rng('shuffle');
S = orth(randn(D,d));
%     X = normc( S*randn(d,N) );
%     O = normc(randn(D,M));
X = S*randn(d,N);
O = randn(D,M);
Xtilde = [X O];
fprintf('Inlier number: %d, outlier number: %d\n', N, M);
Y = normc(Xtilde);
obj = @(b)norm(Y'*b,1);
% fprintf('Fval at the optimal solution: %f\n', obj(S)); this is wrong

for trial = 1:trial_number
    fprintf('Trial number: %d\n', trial);
    
    %% PSGM
    mu_min = 1e-6; % you can inrease mu_min to reduce the time
    maxiter = 200;
    [~, B_PSGM_trial, angle_PSGM_trial, time_PSGM_trial, fval_PSGM_trial] = DPCP_PSGM_optim(Y,c,mu_min,maxiter,S, max_time);

    %% ManPPA
    option.maxiter = 200;
    option.tol = 1e-6;
    option.stepsize = 0.05;  % 0.05
    option.print_inner = 'off';
    option.exact = 0;
    option.c = c;
    option.S = S;
    option.max_time = max_time;
    [B_ManPPA_trial, ~, angle_ManPPA_trial, time_ManPPA_trial, fval_ManPPA_trial] = manppa_DPCP(Y', option);
    

    fprintf('-------ManPPA principal angle: %e\n', abs(asin(norm(B_ManPPA_trial'*S))));

    %% RADMM
    % these five parameters
    option.maxiter = 1000;
    option.max_inner_iter = 1;
    option.rho = 4e1;
    option.gamma = 4e-9;
    option.eta = 1e-4;
    option.max_time = max_time;
    option.tol = 1e-6;
    option.c = c; % this c is the q in the paper
    Y = normc(Xtilde);
    [B_RADMM_trial, angle_RADMM_trial, time_RADMM_trial, fval_RADMM_trial] = radmm_DPCP(Y', option);
    

    fprintf('-------RADMM principal angle: %e\n', abs(asin(norm(B_RADMM_trial'*S))));

    %% IRLS

    tic;
    muilon_J = 10^(-12);
    delta = 10^(-14);
    w = ones(N+M,1);
    %[bo,~] = eigs(Xtilde*Xtilde',c,'SM');

    %B_IRLS = bo;
    %BX = B_IRLS'*Xtilde;
    %J = sum(vecnorm(BX',2,2));
    %angle_IRLS_trial(trial) = abs(asin(norm( vecnorm(B_IRLS'*S,2,2),'inf')));
    %time = tic;time_IRLS(1) = 0;
    J = inf;
    count = 1;
    for i = 1:1500
%         R_X = Xtilde * (w.* Xtilde');
%         %   [U, S, V] = svd(R_X); B = U(:,end);
%         [B_IRLS,~] = eigs(R_X,c,'SM');
        
        [B_IRLS, diag_0] = eig(Y * (w.* Y'));
        [~, ind] = sort(diag(diag_0));
        B_IRLS = B_IRLS(:, ind(1:c));
        
        angle_IRLS_trial(count) = abs(asin(norm(B_IRLS'*S)));
        fval_IRLS_trial(count) = obj(B_IRLS);
        time_IRLS_trial(count) = toc;
        count = count + 1;
        
        BX = B_IRLS'*Y;
        w = 1./max(delta, vecnorm(BX',2,2));
        J_old = J;
        J = sum(vecnorm(BX',2,2));
        % angle_IRLS_trial(trial) = abs(asin(norm( vecnorm(B_IRLS'*S,2,2),'inf')));
        if abs( J - J_old) < 1e-6 || toc > max_time
            fprintf('-------DPCP-IRLS iter: %3d\n', i);
            break;
        end
        %time_IRLS(i) = toc(time);
    end
    
%     %% DPCP-LP
%     % need Gurobi
%     tic;
%     [bo,~] = eigs(Xtilde*Xtilde',c,'SM');
%     pb = [];
%     count = 1;
%     for col = 1:c
%         b = bo(:,col);
%         obj = norm(Xtilde'*b,1);
%         %angle_LP(1) = asin(norm(b'*X));
%         tic;
%         for i = 2:15
%             b = L1_Gurobi(Xtilde,b,pb);
%             b = normc(b);
%             
% %             angle_LP(count) = abs(asin(norm(b'*S)));
% %             time_LP(count) = 0;
% %             count = count + 1;
%             
%             %angle_LP(i) = asin(norm(b(1:d)));
%             %angle_LP(i) = asin(norm(b'*X));
%             obj_old = obj;
%             obj = norm(Xtilde'*b,1);
%             if abs(obj - obj_old)< 1e-6
%                 break;
%             end
%         end
%         pb = [pb b];
%     end
    min_among_all = min([min(fval_PSGM_trial), min(fval_ManPPA_trial(2:end)), min(fval_RADMM_trial), min(fval_IRLS_trial)]);
    l = size(fval_PSGM_trial);
    for i=1:l(2)
        fval_PSGM_trial(i) = fval_PSGM_trial(i) - min_among_all;
    end
    l = size(fval_ManPPA_trial);
    for i=1:l(2)
        fval_ManPPA_trial(i) = fval_ManPPA_trial(i) - min_among_all;
    end
    l = size(fval_RADMM_trial);
    for i=1:l(2)
        fval_RADMM_trial(i) = fval_RADMM_trial(i) - min_among_all;
    end
    l = size(fval_IRLS_trial);
    for i=1:l(2)
        fval_IRLS_trial(i) = fval_IRLS_trial(i) - min_among_all;
    end
    
    if trial == 1
        time_PSGM = time_PSGM_trial; fval_PSGM = fval_PSGM_trial; angle_PSGM = angle_PSGM_trial;
        time_IRLS = time_IRLS_trial; fval_IRLS = fval_IRLS_trial; angle_IRLS = angle_IRLS_trial;
        time_ManPPA = time_ManPPA_trial; fval_ManPPA = fval_ManPPA_trial; angle_ManPPA = angle_ManPPA_trial;
        time_RADMM = time_RADMM_trial; fval_RADMM = fval_RADMM_trial; angle_RADMM = angle_RADMM_trial;
    else
        len = min(length(time_PSGM), length(time_PSGM_trial));
        time_PSGM = time_PSGM(1:len) + time_PSGM_trial(1:len);
        fval_PSGM = fval_PSGM(1:len) + fval_PSGM_trial(1:len);
        angle_PSGM = angle_PSGM(1:len) + angle_PSGM_trial(1:len);
        
        len = min(length(time_IRLS), length(time_IRLS_trial));
        time_IRLS = time_IRLS(1:len) + time_IRLS_trial(1:len);
        fval_IRLS = fval_IRLS(1:len) + fval_IRLS_trial(1:len);
        angle_IRLS = angle_IRLS(1:len) + angle_IRLS_trial(1:len);
        
        len = min(length(time_ManPPA), length(time_ManPPA_trial));
        time_ManPPA = time_ManPPA(1:len) + time_ManPPA_trial(1:len);
        fval_ManPPA = fval_ManPPA(1:len) + fval_ManPPA_trial(1:len);
        angle_ManPPA = angle_ManPPA(1:len) + angle_ManPPA_trial(1:len);
        
        len = min(length(time_RADMM), length(time_RADMM_trial));
        time_RADMM = time_RADMM(1:len) + time_RADMM_trial(1:len);
        fval_RADMM = fval_RADMM(1:len) + fval_RADMM_trial(1:len);
        angle_RADMM = angle_RADMM(1:len) + angle_RADMM_trial(1:len);
    end
end

time_PSGM = time_PSGM / trial_number; 
fval_PSGM = fval_PSGM / trial_number;
angle_PSGM = angle_PSGM / trial_number;

time_IRLS = time_IRLS / trial_number; 
fval_IRLS = fval_IRLS / trial_number;
angle_IRLS = angle_IRLS / trial_number;

time_ManPPA = time_ManPPA / trial_number;
fval_ManPPA = fval_ManPPA / trial_number;
angle_ManPPA = angle_ManPPA / trial_number;

time_RADMM = time_RADMM / trial_number;
fval_RADMM = fval_RADMM / trial_number;
angle_RADMM = angle_RADMM / trial_number;

Time_PSGM = time_PSGM(end); Fval_PSGM = fval_PSGM(end);
Time_IRLS = time_IRLS(end); Fval_IRLS = fval_IRLS(end);
Time_ManPPA = time_ManPPA(end); Fval_ManPPA = fval_ManPPA(end);
Time_RADMM = time_RADMM(end); Fval_RADMM = fval_RADMM(end);

%% plots
plotStyle = {'ro','bo','g-','k:','c-','r:','g:'};
figure1 = figure(1);
clf

loglog(time_ManPPA(end)+eps, angle_ManPPA(end), plotStyle{1},'linewidth',2); legendInfo{1} = ['ManPPA'];
hold on
loglog(time_PSGM(end)+eps, angle_PSGM(end), plotStyle{2},'linewidth',2); legendInfo{2} = ['PSGM'];
hold on 
loglog(time_IRLS+eps, angle_IRLS, plotStyle{3},'linewidth',2); legendInfo{3} = ['IRLS'];
hold on 
loglog(time_RADMM+eps, angle_RADMM, plotStyle{4},'linewidth',2); legendInfo{4} = ['RADMM'];
legend(legendInfo,'Location','Best','FontSize',20);
xlabel('CPU time','FontSize',18); ylabel('\theta','FontSize',18);
%title(['Inlier p_1=' , num2str(N), ', Outlier p_2=', num2str(M), ', dimension of variable: ', num2str(D),'x', num2str(c) ])
filename = "d_" + D + "_p_" + c + "_p1_" + N + "_p2_" + M + "_time_angle.pdf";
saveas(figure1, filename);

figure2 = figure(2);
clf
semilogy(time_ManPPA(end)+eps, fval_ManPPA(end), plotStyle{1},'linewidth',2); legendInfo{1} = ['ManPPA'];
hold on
semilogy(time_PSGM(end)+eps, fval_PSGM(end), plotStyle{2},'linewidth',2); legendInfo{2} = ['PSGM'];
hold on 
semilogy(time_IRLS+eps, fval_IRLS, plotStyle{3},'linewidth',2); legendInfo{3} = ['IRLS'];
hold on 
semilogy(time_RADMM+eps, fval_RADMM, plotStyle{4},'linewidth',2); legendInfo{4} = ['RADMM'];
legend(legendInfo,'Location','Best','FontSize',20);
xlabel('CPU time','interpreter','latex','FontSize',18); ylabel('$\log(f(x)-f^*)$','interpreter','latex','FontSize',18);
%title(['Inlier p_1=' , num2str(N), ', Outlier p_2=', num2str(M), ', dimension of variable: ', num2str(D),'x', num2str(c) ])
filename = "d_" + D + "_p_" + c + "_p1_" + N + "_p2_" + M + "_time_fval.pdf";
saveas(figure2, filename);