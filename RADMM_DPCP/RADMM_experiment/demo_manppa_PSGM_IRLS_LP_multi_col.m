close all; clear all;
%randn('seed',2018);rand('seed',2018)
rng('default');
%% setup
D = 30;
c = 5;
d = D - c;
trial_number = 1;

N = 500; % inlier number
M = 1167;  % outlier number

max_time = 1.0;  % maximum run in seconds

Time_PSGM = 0; Fval_PSGM = 0;
Time_IRLS = 0; Fval_IRLS = 0;
Time_ManPPA = 0; Fval_ManPPA = 0;
Time_RADMM = 0; Fval_RADMM = 0;
err_RADMM = 0;

for trial = 1:trial_number
    rng('shuffle');
    S = orth(randn(D,d));
%     X = normc( S*randn(d,N) );
%     O = normc(randn(D,M));
    X = S*randn(d,N);
    O = randn(D,M);
    Xtilde = [X O];
    fprintf('Inlier number: %d, outlier number: %d, tiral: %d\n', N, M, trial);
    Y = normc(Xtilde);
    obj = @(b)norm(Y'*b,1);
    fprintf('Fval at the optimal solution: %f\n', obj(S));

    %% PSGM
    mu_min = 1e-6; % you can inrease mu_min to reduce the time
    maxiter = 200;
    [~, B_PSGM,angle_PSGM,time_PSGM,fval_PSGM] = DPCP_PSGM_optim(Y,c,mu_min,maxiter,S, max_time);

    %% ManPPA
    option.maxiter = 200;
    option.tol = 1e-6;
    option.stepsize = 0.05;  % 0.05
    option.print_inner = 'off';
    option.exact = 0;
    option.c = c;
    option.S = S;
    option.max_time = max_time;
    [B_ManPPA,~,angle_ManPPA,time_ManPPA, fval_ManPPA] = manppa_DPCP(Y', option);

    fprintf('-------ManPPA principal angle: %e\n', abs(asin(norm(B_ManPPA'*S))));

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
    [B_RADMM,angle_RADMM,time_RADMM,fval_RADMM] = radmm_DPCP(Y', option);

    fprintf('-------RADMM principal angle: %e\n', abs(asin(norm(B_RADMM'*S))));

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
        
        angle_IRLS(count) = abs(asin(norm(B_IRLS'*S)));
        fval_IRLS(count) = obj(B_IRLS);
        time_IRLS(count) = toc;
        count = count + 1;
        
        BX = B_IRLS'*Y;
        w = 1./max(delta, vecnorm(BX',2,2));
        J_old = J;
        J = sum(vecnorm(BX',2,2));
        angle_IRLS_trial(trial) = abs(asin(norm( vecnorm(B_IRLS'*S,2,2),'inf')));
        if abs( J - J_old) < 1e-6 || toc > max_time
            fprintf('-------DPCP-IRLS iter: %3d\n', i);
            break;
        end
        %time_IRLS(i) = toc(time);
    end
    
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
    legend(legendInfo,'Location','Best');
    xlabel('CPU time'); ylabel('\theta');
    title(['Inlier p_1=' , num2str(N), ', Outlier p_2=', num2str(M), ', dimension of variable: ', num2str(D),'x', num2str(c) ])
    filename = "d_" + D + "_p_" + c + "_p1_" + N + "_p2_" + M + "_time_angle.pdf";
    % saveas(figure1, filename);
    
    figure2 = figure(2);
    clf
    semilogy(time_ManPPA(end)+eps, fval_ManPPA(end), plotStyle{1},'linewidth',2); legendInfo{1} = ['ManPPA'];
    hold on
    semilogy(time_PSGM(end)+eps, fval_PSGM(end), plotStyle{2},'linewidth',2); legendInfo{2} = ['PSGM'];
    hold on 
    semilogy(time_IRLS+eps, fval_IRLS, plotStyle{3},'linewidth',2); legendInfo{3} = ['IRLS'];
    hold on 
    semilogy(time_RADMM+eps, fval_RADMM, plotStyle{4},'linewidth',2); legendInfo{4} = ['RADMM'];
    legend(legendInfo,'Location','Best');
    xlabel('CPU time'); ylabel('Fval');
    title(['Inlier p_1=' , num2str(N), ', Outlier p_2=', num2str(M), ', dimension of variable: ', num2str(D),'x', num2str(c) ])
    filename = "d_" + D + "_p_" + c + "_p1_" + N + "_p2_" + M + "_time_fval.pdf";
    % saveas(figure2, filename);
    
    Time_PSGM = Time_PSGM + time_PSGM(end); Fval_PSGM = Fval_PSGM + fval_PSGM(end);
    Time_IRLS = Time_IRLS + time_IRLS(end); Fval_IRLS = Fval_IRLS + fval_IRLS(end);
    Time_ManPPA = Time_ManPPA + time_ManPPA(end); Fval_ManPPA = Fval_ManPPA + fval_ManPPA(end);
    Time_RADMM = Time_RADMM + time_RADMM(end); Fval_RADMM = Fval_RADMM + fval_RADMM(end);
    
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
end

Time_PSGM = Time_PSGM / trial_number; Fval_PSGM = Fval_PSGM / trial_number;
Time_IRLS = Time_IRLS / trial_number; Fval_IRLS = Fval_IRLS / trial_number;
Time_ManPPA = Time_ManPPA / trial_number; Fval_ManPPA = Fval_ManPPA / trial_number;
Time_RADMM = Time_RADMM / trial_number; Fval_RADMM = Fval_RADMM / trial_number;

% %% plot the results
% fontsize = 10;
% plotStyle = {'r-','b-','g-','k:','c-','r:','g:'};
% figure
% semilogy(0:length(angle_ManPPA)-1,angle_ManPPA,plotStyle{1},'linewidth',2);
% legendInfo{1} = ['ManPPA'];
% hold on
% semilogy(0:length(angle_PSGM)-1,angle_PSGM,plotStyle{2},'linewidth',2);
% legendInfo{2} = ['PSGM-{MBLS}'];
% semilogy(0:length(angle_IRLS)-1,angle_IRLS,plotStyle{3},'linewidth',2);
% legendInfo{3} = ['IRLS'];
% semilogy(0:length(angle_RADMM)-1,angle_RADMM,plotStyle{4},'linewidth',2);
% legendInfo{4} = ['RADMM'];
% % semilogy(0:length(angle_SPPA)-1,angle_SPPA,plotStyle{5},'linewidth',2);
% % legendInfo{5} = ['StManPPA-' num2str(option.stepsize)];
% % T_LP = length(find(angle_LP>0)); %angle_LP(T_LP+1) = 1e-13;
% % semilogy(0:length(angle_LP)-1,angle_LP,plotStyle{6},'linewidth',2);
% % legendInfo{6} = ['ALP'];
% %semilogy(0:length(angle_ManPPA1)-1,angle_ManPPA1,plotStyle{7},'linewidth',2);
% %legendInfo{7} = ['ManPPA-exact'];
% ylim([min([min(angle_ManPPA),min(angle_IRLS)])/10, pi/2])
% xlim([0 60])
% legend(legendInfo,'Location','Best')
% xlabel('iteration','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
% ylabel('$\theta$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex','rot',0);
% set(gca,'FontSize',16);
% set(legend,'color','none');
% 
% % set(gca,'YDir','normal')
% % set(gca, ...
% %     'LineWidth' , 2                     , ...
% %     'FontSize'  , fontsize              , ...
% %     'FontName'  , 'Times New Roman'         );
% % set(gcf, 'Color', 'white');
% saveas(gcf,'DPCP-rand-iter','epsc')
% %%%
% figure
% loglog(time_ManPPA,angle_ManPPA,plotStyle{1},'linewidth',2);
% hold on
% loglog(time_PSGM,angle_PSGM,plotStyle{2},'linewidth',2);
% hold on
% loglog(time_IRLS,angle_IRLS,plotStyle{3},'linewidth',2);
% hold on
% loglog(time_RADMM,angle_RADMM,plotStyle{4},'linewidth',2);
% hold on
% % loglog(time_LP,angle_LP,plotStyle{5},'linewidth',2);
% % hold on
% % loglog(time_SPPA,angle_SPPA,plotStyle{6},'linewidth',2);
% % hold on
% ylim([min([min(angle_ManPPA),min(angle_IRLS)])/10, pi/2])
% xlim([0 60])
% % xlim([0 max(time_LP((angle_LP(1:T_LP)<pi/2)))])
% legend(legendInfo,'Location','Best')
% xlabel('time','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
% ylabel('$\theta$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex','rot',0);
% set(gca,'FontSize',16);
% set(legend,'color','none');
% 
% % set(gca,'YDir','normal')
% % set(gca, ...
% %     'LineWidth' , 2                     , ...
% %     'FontSize'  , fontsize              , ...
% %     'Xtick'    , [1e-3 1e-2 1e-1 1 10],...
% %     'FontName'  , 'Times New Roman'         );
% % set(gcf, 'Color', 'white');
% saveas(gcf,'DPCP-rand-CPU','epsc')