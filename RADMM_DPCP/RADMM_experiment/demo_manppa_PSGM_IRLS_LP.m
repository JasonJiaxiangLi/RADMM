close all; clear all;
%randn('seed',2018);rand('seed',2018)
rng('default');
rng('shuffle')
%% setup
D = 30;  
N = 1000;
d = 29; c = D - d;
ratio = 1 ./ (1 ./ 0.7 - 1);
noise_level = 0.;
M = round(N * ratio);
S = orth(randn(D,d)); % ground truth random
%S = eye(D,d);     % ground truth Identity

X = normc( S*randn(d,N)  + noise_level*randn(D,N));
O = normc(randn(D,M));
Xtilde = [X O];
obj = @(b) norm(Xtilde'*b,1);
% initialization
[bo,~] = eigs(Xtilde*Xtilde',c,'SM');

maxiter = 200;


% pre run Manppa
option.maxiter = 5;
option.tol = 1e-9;
option.stepsize = 0.1;
option.print_inner = 'off';
option.c = c;
option.exact = 0;
option.S = S;

Y = normc(Xtilde);
option.phi_init = bo;
[B_ManPPA,~,~,~,~,~,angle_ManPPA,tManPPA] = manppa_DPCP1(Y', option);
%% ManPPA - inexact
option.maxiter = 100;
option.tol = 1e-9;
option.stepsize = 0.1;
option.print_inner = 'off';
option.c = c;
option.exact = 0;
option.S = S;

Y = normc(Xtilde);
option.phi_init = bo;
[B_ManPPA,~,~,~,~,~,angle_ManPPA,tManPPA] = manppa_DPCP1(Y', option);

%% ManPPA-exact
% option.maxiter = 100;
% option.tol = 1e-9;
% option.stepsize = 0.1;
% option.print_inner = 'off';
% option.c = c;
% option.exact = 1;
% option.S = S;
% 
% Y = normc(Xtilde);
% option.phi_init = bo;
% [B_ManPPA1,~,~,~,~,~,angle_ManPPA1,tManPPA1] = manppa_DPCP1(Y', option);
%% PSGD with modified line search
%pre_run PSGM
mu_ls = 0.01; alpha = 0.001; beta_ls = 0.5;
b = bo;
angle_line(1) = asin(norm(b'*S));
%angle_line(1) = asin(norm(b'*X));

time = tic;time_line(1) = 0;
obj_old = obj(b);
mu_threshold = 1e-15;  % here mu_threshold is used to reduce the total time,
%the smaller them are, closer b to the complematry subspace, but more time needed
mu = mu_ls;
for i = 2:3
   % grad = sum( repmat(sign(b'*Xtilde),D,1).*Xtilde, 2);
    grad = Xtilde*( sign(Xtilde'*b));
    grad_norm = norm(grad)^2;
    while (obj( normc(b - mu*grad) )> obj_old - alpha*mu*grad_norm)&& mu>mu_threshold
        mu = mu*beta_ls;
    end
    b = normc(b - mu*grad);
    obj_old = obj(b);
    %angle_line(i) = asin(norm(b(1:d)));
    angle_line(i) = asin(norm(b'*S));
    time_line(i) = toc(time);
end
% 
mu_ls = 0.01; alpha = 0.001; beta_ls = 0.5;
b = bo;
angle_line(1) = asin(norm(b'*S));
%angle_line(1) = asin(norm(b'*X));

time = tic;time_line(1) = 0;
obj_old = obj(b);
mu_threshold = 1e-15;  % here mu_threshold is used to reduce the total time,
%the smaller them are, closer b to the complematry subspace, but more time needed
mu = mu_ls;
for i = 2:maxiter
   % grad = sum( repmat(sign(b'*Xtilde),D,1).*Xtilde, 2);
    grad = Xtilde*( sign(Xtilde'*b));
    grad_norm = norm(grad)^2;
    while (obj( normc(b - mu*grad) )> obj_old - alpha*mu*grad_norm)&& mu>mu_threshold
        mu = mu*beta_ls;
    end
    b = normc(b - mu*grad);
    obj_old = obj(b);
    %angle_line(i) = asin(norm(b(1:d)));
    angle_line(i) = asin(norm(b'*S));
    time_line(i) = toc(time);
end

%% IRLS
% DPCP_IRLS parameters
%pre_run IRLS
muilon_J = 10^(-12);
delta = 10^(-14);
w = ones(N+M,1);
B = bo;
angle_IRLS(1) = asin(norm(B'*S));
%angle_IRLS(1) = asin(norm(B'*X));
time = tic;time_IRLS(1) = 0;
J = Inf;
for i = 2:3
    R_X = Xtilde *(w.* Xtilde');
    %   [U, S, V] = svd(R_X); B = U(:,end);
    [B,~] = eigs(R_X,c,'SM');
    %     for j = 1 : N+M
    %         w(j) = 1/max(delta,norm(B'*Xtilde(:,j)));
    %     end
    %w = 1./max(delta, abs(Xtilde'*B));
    w = 1./max(delta, vecnorm(Xtilde'*B,2,2));
    angle_IRLS(i) = asin(norm(B'*S));
    %angle_IRLS(i) = asin(norm(B'*X));
    time_IRLS(i) = toc(time);
    J_old = J;
    BX = B'*Xtilde;
    J = sum(vecnorm(BX',2,2));
    if abs( J - J_old) < 1e-11
        fprintf('-------DPCP-IRLS iter: %3d\n', i);
        break;
    end
end
w = ones(N+M,1);
B = bo;
angle_IRLS(1) = asin(norm(B'*S));
%angle_IRLS(1) = asin(norm(B'*X));
time = tic;time_IRLS(1) = 0;
J = Inf;
for i = 2:maxiter
    R_X = Xtilde *(w.* Xtilde');
    %   [U, S, V] = svd(R_X); B = U(:,end);
    [B,~] = eigs(R_X,c,'SM');
    %     for j = 1 : N+M
    %         w(j) = 1/max(delta,norm(B'*Xtilde(:,j)));
    %     end
    %w = 1./max(delta, abs(Xtilde'*B));
    w = 1./max(delta, vecnorm(Xtilde'*B,2,2));
    angle_IRLS(i) = asin(norm(B'*S));
    %angle_IRLS(i) = asin(norm(B'*X));
    time_IRLS(i) = toc(time);
    J_old = J;
    BX = B'*Xtilde;
    J = sum(vecnorm(BX',2,2));
    if abs( J - J_old) < 1e-11
        fprintf('-------DPCP-IRLS iter: %3d\n', i);
        break;
    end
end

%% LP
%pre_run DPCP LP
b = bo;
pb = [];
for col = 1:c
    angle_LP(1) = asin(norm(b'*S));
    obj_lp = norm(Xtilde'*b,1);
    time = tic;
    time_LP(1) = 0;
    for i = 2:3
        b = L1_Gurobi(Xtilde,b,pb);
        b = normc(b);
        angle_LP(i) = asin(norm(b'*S));
        %angle_LP(i) = asin(norm(b'*X));
        time_LP(i) = toc(time);
        obj_old = obj_lp;
        obj_lp = norm(Xtilde'*b,1);
        if abs(obj_lp - obj_old)< 1e-6
            fprintf('-------DPCP-LP iter: %3d\n', i);
            break;
        end
    end
    pb = [pb b];
end

b = bo;
pb = [];

for col = 1:c
    angle_LP(1) = asin(norm(b'*S));
    obj_lp = norm(Xtilde'*b,1);
    time = tic;
    time_LP(1) = 0;
    for i = 2:15
        b = L1_Gurobi(Xtilde,b,pb);
        b = normc(b);
        angle_LP(i) = asin(norm(b'*S));
        %angle_LP(i) = asin(norm(b'*X));
        time_LP(i) = toc(time);
        obj_old = obj_lp;
        obj_lp = norm(Xtilde'*b,1);
        if abs(obj_lp - obj_old)< 1e-6
            fprintf('-------DPCP-LP iter: %3d\n', i);
            break;
        end
    end
    pb = [pb b];
end



%% SPPA
option.tol = 1e-12;
option.stepsize = 0.6;
option.S = S;
[q_manpg_sppa, ~, ~, time_SPPA,angle_SPPA] = manppa_sppa_DPCP(Y,option);
[q_manpg_sppa, ~, ~, time_SPPA,angle_SPPA] = manppa_sppa_DPCP(Y,option);
% semilogy(manpg_sppa_F-min(manpg_sppa_F))

%% RADMM
rho = 1e5; eta = 1e-5; gamma = 1e-8;
b = bo;
angle_radmm(1) = asin(norm(b'*S));
obj_val_radmm(1) = obj(b);
time = tic;time_radmm(1) = 0;
obj_old = obj(b);
lambda = zeros(N + M, 1); z = Xtilde.'*b;
maxiter = 200;
for iter = 2:maxiter
    % X step: a gradient step
    for i=1:1
        gx = Xtilde*lambda + rho*Xtilde*(Xtilde.'*b - z);
        rgx = gx - (b.'*gx)*b;
        b = (b - eta*rgx) / norm(b - eta*rgx);
        % fprintf('inner iter: %d, X step subgrad:%f\n', i, norm(rgx));
    end

    % Z step (also update Y)
    y = wthresh(Xtilde.'*b+lambda/rho,'s', (1+rho*gamma)/rho);
    z = (y/gamma + lambda + rho*Xtilde.'*b) / (1/gamma + rho);

    % Lambda step
    lambda = lambda + rho*(Xtilde.'*b - z);

    % update gamma
    %     if norm(Z-Y,'fro')>= 0.9*dist_ZY(iter-1) && gamma >= eps
    %         gamma = gamma/2;
    %     end

    %     if norm(X-Z,'fro')>= 0.9*dist_XZ(iter-1) && rho <=1e6
    %         rho = rho*2;
    %     end

    % Value update
    time_radmm(iter) = toc(time);
    obj_val_radmm(iter) = obj(b);
    angle_radmm(iter) = asin(norm(b'*S));
    
    fprintf('ADMM iter: %d, XT*b - z: %f, fval: %f, angle: %f\n', iter, norm(Xtilde.'*b - z), obj_val_radmm(iter), angle_radmm(iter));
    
    if abs(obj(b) - obj_old) <= 1e-6 && norm(Xtilde.'*b - z) <= 1e-6
        break
    end
    obj_old = obj(b);
end

%% plot the results
fontsize = 10;
plotStyle = {'r-','b-','g-','k:','c-','r:','g:'};
figure
semilogy(0:length(angle_ManPPA)-1,angle_ManPPA,plotStyle{1},'linewidth',2);
legendInfo{1} = ['ManPPA'];
hold on
semilogy(0:length(angle_line)-1,angle_line,plotStyle{2},'linewidth',2);
legendInfo{2} = ['PSGM-{MBLS}'];
semilogy(0:length(angle_IRLS)-1,angle_IRLS,plotStyle{3},'linewidth',2);
legendInfo{3} = ['IRLS'];
T_LP = length(find(angle_LP>0)); %angle_LP(T_LP+1) = 1e-13;
semilogy(0:length(angle_LP)-1,angle_LP,plotStyle{4},'linewidth',2);
legendInfo{4} = ['ALP'];
semilogy(0:length(angle_SPPA)-1,angle_SPPA,plotStyle{5},'linewidth',2);
legendInfo{5} = ['StManPPA-' num2str(option.stepsize)];
semilogy(0:length(angle_radmm)-1,angle_radmm,plotStyle{6},'linewidth',2);
legendInfo{6} = ['RADMM'];
%semilogy(0:length(angle_ManPPA1)-1,angle_ManPPA1,plotStyle{6},'linewidth',2);
%legendInfo{6} = ['ManPPA-exact'];
ylim([min([min(angle_ManPPA),min(angle_line)])/10, pi/2])
xlim([0 60])
legend(legendInfo,'Location','Best')
xlabel('iteration','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
ylabel('$\theta$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex','rot',0);
set(gca,'FontSize',16);
set(legend,'color','none');

% set(gca,'YDir','normal')
% set(gca, ...
%     'LineWidth' , 2                     , ...
%     'FontSize'  , fontsize              , ...
%     'FontName'  , 'Times New Roman'         );
% set(gcf, 'Color', 'white');
saveas(gcf,'DPCP-rand-iter','epsc')
%%%
figure
loglog(tManPPA,angle_ManPPA,plotStyle{1},'linewidth',2);
hold on
loglog(time_line,angle_line,plotStyle{2},'linewidth',2);
hold on

loglog(time_IRLS,angle_IRLS,plotStyle{3},'linewidth',2);
hold on
loglog(time_LP,angle_LP,plotStyle{4},'linewidth',2);
hold on
loglog(time_SPPA,angle_SPPA,plotStyle{5},'linewidth',2);
hold on
loglog(time_radmm,angle_radmm,plotStyle{6},'linewidth',2);
hold on
ylim([min([min(angle_ManPPA),min(angle_line)])/10, pi/2])
xlim([0 max(time_LP((angle_LP(1:T_LP)<pi/2)))])
legend(legendInfo,'Location','Best')
xlabel('time','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
ylabel('$\theta$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex','rot',0);
set(gca,'FontSize',16);
set(legend,'color','none');

% set(gca,'YDir','normal')
% set(gca, ...
%     'LineWidth' , 2                     , ...
%     'FontSize'  , fontsize              , ...
%     'Xtick'    , [1e-3 1e-2 1e-1 1 10],...
%     'FontName'  , 'Times New Roman'         );
% set(gcf, 'Color', 'white');
saveas(gcf,'DPCP-rand-CPU','epsc')
