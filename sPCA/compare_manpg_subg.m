%% RADMM with Moreau envelope
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
% n_list = [300, 500]; p_list = [50, 100];
n_list = 300; p_list = 100;
% mu_list = [0.5, 0.7, 1];
mu_list = 1;
for n=n_list
    for p=p_list
        for mu=mu_list
            disp("test on n="+ n +" p="+ p+" mu="+mu);
            K = n;
            A = randn(n, K); A = orth(A);
            S = diag(abs(randn(K,1)));
            H = A*S*A.';
            % mu = 0.5;
            f = @(X) -0.5*trace(X.'*H*X);
            nabla_f = @(X) -H*X;
            g = @(Y) mu*sum(sum(abs(Y)));
            g_gamma = @(Z,gamma) mu*(g(wthresh(Z,'s',gamma))+1/(2*gamma)*norm(wthresh(Z,'s',gamma) - Z,'fro')^2);
            F = @(X) f(X) + g(X);
            sub_F = @(X) - H*X + mu*sign(X);

            %% Algorithm
            % initialization in Stiefel manifold
            X = randn(n, p);
            X = orth(X);
            Y = X; Z = X; U = X;
            Lambda = zeros(size(X));
            eta = 1e-2; gamma = 1e-8; N = 1000; rho = 100;

            % Parameters for ManPG subproblem
            L = abs(eigs(full(H),1)); % Lipschitz constant
            t = 1/L;
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
            tol = 1e-8*n*p;

            Lag = @(X,Y,Lambda,gamma,rho) f(X) + mu*g(Y) + trace(Lambda.'*(X-Y)) + rho/2*norm(X-Y)^2;
            L_M = @(X,Z,Lambda,gamma,rho) f(X) + mu*g_gamma(Z,gamma) + trace(Lambda.'*(X-Z)) + rho/2*norm(X-Z)^2;

            iter = 3;
            F_val(iter) = F(X); F_val_manpg(iter) = F(X);
            F_val_subg(iter) = F(X);
            L_val(iter) = Lag(X,Y,Lambda,gamma,rho);
            MEL_val(iter) = L_M(X,Z,Lambda,gamma,rho);
            dist_XY(iter) = norm(X-Y,'fro'); dist_XZ(iter) = norm(X-Z,'fro'); dist_ZY(iter) = norm(Z-Y,'fro');
            norm_subg_F(iter) = norm(proj(X, sub_F(X)),'fro'); 

            avg = 1;
            F_val_manpg_avg = zeros([1,1000]);
            F_val_subg_avg = zeros([1,1000]);
            F_val_avg = zeros([1,1000]);
            L_val_avg = zeros([1,1000]);
            dist_XY_avg = zeros([1,1000]);
            dist_ZY_avg = zeros([1,1000]);
            dist_XZ_avg = zeros([1,1000]);
            cpu_time_manpg = zeros([avg,1000]);
            cpu_time_admm = zeros([avg,1000]);
            cpu_time_subg = zeros([avg, 1000]);
            sparse_X = zeros([1, avg]);
            error_Y = zeros([1, avg]);
            sparse_U = zeros([1, avg]);
            sparse_W = zeros([1, avg]);


            iter1 = N; iter2 = N; iter3 = N;
            for k = 1:avg
                if k == avg
                    disp("Total repitition " + k);
                end
                X = randn(n, p);
                X = orth(X);
                Y = X; Z = X; U = X; W = X;
                Lambda = zeros(size(X));
                eta = 1e-2; gamma = 1e-8; N = 1000; rho = 100;

                %% ManPG
                F_val_manpg_avg(1) = F_val_manpg_avg(1)+F(X);
                for iter=2:N
                    manpg_start = tic;
                    neg_pg = -H*U;
                    if alpha < t_min || num_inexact > 10
                        inner_tol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
                    else
                        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
                    end

                    % The subproblem
                    %semi_newton = tic;
                    if iter == 2
                         [ PU,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,U,t,U + t*neg_pg,nu*t,inner_tol,prox_fun,inner_iter,zeros(p),Dn,pDn);
                        %      [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
                    else
                         [ PU,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,U,t,U + t*neg_pg,nu*t,inner_tol,prox_fun,inner_iter,Lam_x,Dn,pDn);
                        %     [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
                    end
                    %semi_newton_end = toc(semi_newton);
                    %cpu_time_newton(iter) = cpu_time_newton(iter)+semi_newton_end;
                    if in_flag == 1   % subprolem not exact.
                        inner_flag = 1 + inner_flag;
                    end

                    V = PU-U; % The V solved from SSN

                    % projection onto the Stiefel manifold
                    [T, SIGMA, S] = svd(PU'*PU);   SIGMA =diag(SIGMA);    U_temp = PU*(T*diag(sqrt(1./SIGMA))*S');

                    f_trial = f(U_temp);
                    F_trial = f_trial + g(U_temp);   normV=norm(V,'fro');

                %     if  normD < tol 
                %         break;
                %     end

                %     %%% linesearch
                %     alpha_x = 1;
                %     while F_trial >= F_val(iter-1)-0.5/t*alpha_x*normV^2
                %         alpha_x = 0.5*alpha_x;
                %         linesearch_flag = 1;
                %         num_linesearch_x = num_linesearch_x + 1;
                %         if alpha_x < t_min
                %             num_inexact_x = num_inexact_x + 1;
                %             break;
                %         end
                %         PX = X+alpha_x*V;
                %         % projection onto the Stiefel manifold
                %         [U, SIGMA, S] = svd(PX'*PX);   SIGMA =diag(SIGMA);   X_temp = PX*(U*diag(sqrt(1./SIGMA))*S');
                %         f_trial = f(X_temp,H);
                %         F_trial = f_trial + lambda*h(X_temp);
                %     end
                %     X = X_temp; step_size_x(iter) = alpha_x;
                %     F_val(iter) = F_trial;
                %     norm_x(iter) = normV;

                    %%% Without linesearch
                    PU = U+alpha*V;
                    % projection onto the Stiefel manifold
                    [T, SIGMA, S] = svd(PU'*PU);   SIGMA =diag(SIGMA);   U_temp = PU*(T*diag(sqrt(1./SIGMA))*S');
                    U = U_temp; % update
                    elapsed_time_manpg = toc(manpg_start);
                    F_val_manpg(iter) = F(U);
                    
                    if abs(F_val_manpg(iter) - F_val_manpg(iter-1)) <= 1e-8
                        break
                    end
                    
                    norm_x(iter) = normV;
                    norm_subg_ManPG(iter) = norm(proj(U, sub_F(U)),'fro'); 

                    cpu_time_manpg(k,iter) = cpu_time_manpg(k,iter) + elapsed_time_manpg;
                    if iter < 1000
                        cpu_time_manpg(k,iter+1) = cpu_time_manpg(k,iter);
                    end
                    
                end
                iter1 = min(iter, iter1);
                

                %% RADMM
                for iter=2:N
                    admm_start = tic;
                    % X step: a gradient step
                    for i=1:1
                        gx = -H*X + Lambda + rho*(X - Z);
                        rgx = proj(X, gx);
                        X = retr(X, -(eta)*rgx);
                        %fprintf('inner iter: %d, X step subgrad:%f\n', i, norm(rgx));
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
                    elapsed_time_admm = toc(admm_start);

                    % Value update
                    F_val(iter) = F(X);
                    if abs(F_val(iter) - F_val(iter-1)) <= 1e-8
                        break
                    end
                    L_val(iter) = Lag(X,Y,Lambda,gamma,rho);
                    MEL_val(iter) = L_M(X,Z,Lambda,gamma,rho);
                    dist_XY(iter) = norm(X-Y,'fro'); 
                    dist_XZ(iter) = norm(X-Z,'fro'); 
                    dist_ZY(iter) = norm(Z-Y,'fro');
                    norm_subg_F(iter) = norm(proj(X, sub_F(X)),'fro'); 

                    cpu_time_admm(k,iter) = cpu_time_admm(k,iter) + elapsed_time_admm;
                    if iter < 1000
                        cpu_time_admm(k,iter+1) = cpu_time_admm(k,iter);
                    end

                    % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
                end
                iter2 = min(iter, iter2);
                L_val_avg(1:iter-1) = L_val_avg(1:iter-1) + L_val(1:iter-1);
                
                dist_XY_avg(1:iter-1) = dist_XY_avg(1:iter-1) + dist_XY(1:iter-1);
                dist_XZ_avg(1:iter-1) = dist_XZ_avg(1:iter-1) + dist_XZ(1:iter-1);
                dist_ZY_avg(1:iter-1) = dist_ZY_avg(1:iter-1) + dist_ZY(1:iter-1);

                %% Riemannian Subgrad method
                F_val_subg_avg(1) = F_val_subg_avg(1) + F(W);
                eta = 1e-2;
                for iter=2:N

                    subg_start = tic;
                    neg_g = -nabla_f(W)-mu*sign(W); % negative subgradient
                    neg_rg = proj(W, neg_g); % projected onto the tangent space
                    W = retr(W, eta*neg_rg);
                    elapsed_time_subg = toc(subg_start);
                    F_val_subg(iter) = F(W);
                    
                    if abs(F_val_subg(iter) - F_val_subg(iter-1)) <= 1e-8
                        break
                    end
                    norm_subgrad(iter) = norm(neg_rg,'fro');
                    cpu_time_subg(k,iter) = cpu_time_subg(k,iter) + elapsed_time_subg;
                    if iter < 1000
                        cpu_time_subg(k,iter+1) = cpu_time_subg(k,iter);
                    end
                end
                iter3 = min(iter, iter3);
                

                sparse_X(k) = sum(sum(abs(Y) <= 1e-8))/(n*p);
                error_Y(k) = norm(Y.'*Y - eye(p), 'fro');
                sparse_U(k) = sum(sum(abs(U) <= 1e-8))/(n*p);
                sparse_W(k) = sum(sum(abs(W) <= 1e-8))/(n*p);
                % sum(sum(Z)), error: norm(Z.'*Z, 'fro')
                
                %% sum all the averages
                min_among_all = min([min(F_val_manpg(2:end)), min(F_val(2:end)), min(F_val_subg(2:end))]) - eps;
                l = size(F_val_manpg);
                for i=1:l(2)
                    F_val_manpg(i) = F_val_manpg(i) - min_among_all;
                end
                l = size(F_val);
                for i=1:l(2)
                    F_val(i) = F_val(i) - min_among_all;
                end
                l = size(F_val_subg);
                for i=1:l(2)
                    F_val_subg(i) = F_val_subg(i) - min_among_all;
                end
                F_val_manpg_avg(1:iter1) = F_val_manpg_avg(1:iter1)+ F_val_manpg(1:iter1);
                F_val_avg(1:iter2) = F_val_avg(1:iter2) + F_val(1:iter2);
                F_val_subg_avg(1:iter3) = F_val_subg_avg(1:iter3) + F_val_subg(1:iter3);
            end 
            F_val_avg= (F_val_avg/avg);
            L_val_avg = (L_val_avg/avg);
            dist_XY_avg = (dist_XY_avg/avg);
            dist_XZ_avg = (dist_XZ_avg/avg);
            dist_ZY_avg = (dist_ZY_avg/avg);
            F_val_manpg_avg = (F_val_manpg_avg/avg);
            F_val_subg_avg = (F_val_subg_avg/avg);
            %cpu_time_newton = cpu_time_newton/avg

            cpu_time_admm = sum(cpu_time_admm,1)/avg;
            cpu_time_manpg = sum(cpu_time_manpg,1)/avg;
            cpu_time_subg = sum(cpu_time_subg,1)/avg;

            av_sparse_x = sum(sparse_X)/avg;
            av_sparse_u = sum(sparse_U)/avg;
            av_sparse_w = sum(sparse_W)/avg;
            
            av_error_y = sum(error_Y)/avg;

            disp("sparisty for rgrad, manpg and RADMM: ")

            disp([av_sparse_w, av_sparse_u, av_sparse_x])
            
            disp("error of RADMM: ")
            
            disp(av_error_y)
            
            disp("CPU time for rgrad, manpg and RADMM: ")
            
            disp([cpu_time_subg(iter3 - 1), cpu_time_manpg(iter1 - 1), cpu_time_admm(iter2 - 1)]);
            
            disp("function value for output rgrad, manpg and RADMM: ")

            disp([F_val_subg_avg(iter3 - 1), F_val_manpg_avg(iter1 - 1), F_val_avg(iter2 - 1)]);

            %{
            fileID = fopen('myfile.txt','w');
            fprintf(fileID, "manpg\n");
            for a = 1:1000
                fprintf(fileID, "%f ", cpu_time_manpg(1,a));
            end 
            fprintf(fileID, "\n admm\n");
            for a = 1:1000
                fprintf(fileID, "%f ", cpu_time_admm(1,a));
            end

            cpu_time_admm = cpu_time_admm;
            cpu_time_manpg = cpu_time_admm - x;
            %}
            %% Plots
%             figure1 = figure(1);
%             clf
%             plot(F_val_avg); hold on;
%             plot(L_val_avg); hold on;
%             %plot(norm_subg_F,'-*'); hold on;
%             legend('Function value at X', 'Lagrangian value');
%             % filename = "grid_search_plots/n_" + n + "_p_" + p + "_mu_" + mu + "_lval.pdf";
%             % saveas(figure1, filename);
% 
% 
%             figure2 = figure(2);
%             clf
%             semilogy(dist_XZ_avg); hold on;
%             semilogy(dist_ZY_avg); hold on;
%             % plot(norm_subg_F,'-*'); hold on;
%             legend('distance of X to Z', 'distance of Z to Y');
%             % filename = "grid_search_plots/n_" + n + "_p_" + p + "_mu_" + mu + "_distval.pdf";
%             % saveas(figure2, filename);


            % figure3 = figure(3);
            % clf
            % semilogy(F_val_avg); hold on;
            % semilogy(F_val_manpg_avg); hold on;
            % title('function values');
            % legend('RADMM', 'ManPG');
            % saveas(figure3, 'grid_search_plots/n_500_p_50_fval.pdf');


            figure4 = figure(4);
            clf
%             semilogy(cpu_time_subg, F_val_subg_avg); hold on;
%             semilogy(cpu_time_manpg, F_val_manpg_avg); hold on;
%             semilogy(cpu_time_admm, F_val_avg); hold on;
            semilogy(cpu_time_subg(2:iter3), F_val_subg_avg(2:iter3),'LineWidth',2); hold on;
            semilogy(cpu_time_manpg(2:iter1), F_val_manpg_avg(2:iter1),'LineWidth',2); hold on;
            semilogy(cpu_time_admm(2:iter2), F_val_avg(2:iter2),'LineWidth',2); hold on;
            xlabel('CPU time','interpreter','latex','FontSize',18); ylabel('$f(x)-f^*$','interpreter','latex','FontSize',18);
            legend('RSG', 'ManPG', 'RADMM');
            legend('Location','best','FontSize',20);
            filename = "grid_search_plots/n_" + n + "_p_" + p + "_mu_" + mu + "_time_fval.pdf";
            saveas(figure4, filename);
            % break;
        end

    end
end

