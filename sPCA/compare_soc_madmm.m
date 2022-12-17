%% RADMM with Moreau envelope
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementation of the Moreau-envelope Riemannian ADMM
% Test on the sPCA problem
% Problem: sPCA, min -1/2*tr(X^THX)+\mu*\|X\|_1=f(X)+h(X)
% Manifold: Stiefel manifold St(n, p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear

%% Problem Generating
n_list = [300, 500]; p_list = [50, 100];
% n_list = 300; p_list = 50;
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
            N = 1000;
            iter = 1;
            avg = 1;
            F_val_soc_avg = zeros([1,1000]);
            F_val_madmm_avg = zeros([1,1000]);
            F_val_radmm_avg = zeros([1,1000]);
            cpu_time_soc = zeros([avg,1000]); cpu_time_soc(1) = eps;
            cpu_time_madmm = zeros([avg,1000]); cpu_time_madmm(1) = eps;
            cpu_time_radmm = zeros([avg, 1000]); cpu_time_radmm(1) = eps;
            vio_soc_avg = zeros([1,1000]);
            vio_madmm_avg = zeros([1,1000]);
            vio_radmm_avg = zeros([1,1000]);
            
            sparse_soc = zeros([1, avg]);
            error_soc = zeros([1, avg]);
            sparse_madmm = zeros([1, avg]);
            error_madmm = zeros([1, avg]);
            sparse_radmm = zeros([1, avg]);
            error_radmm = zeros([1, avg]);

            iter0 = N; iter1 = N; iter2 = N;
            disp("Total repitition " + avg);
            for k = 1:avg
                % random initialize
                X0 = randn(n, p);
                X0 = orth(X0);
                F_val_soc_avg(1) = F_val_soc_avg(1) + F(X0);
                F_val_madmm_avg(1) = F_val_madmm_avg(1) + F(X0);
                F_val_radmm_avg(1) = F_val_radmm_avg(1) + F(X0);
                
                %% SOC
                X = X0; Y = X0;
                Lambda = zeros(size(X));
                eta = 1e-2; N0 = 100; rho = 5e1;
                
                for iter=2:N0
                    temp_F = @(X) -0.5*trace(X.'*H*X) + mu*sum(sum(abs(X))) + rho / 2 * norm(X - Y + Lambda, 'fro')^2;
                    admm_start = tic;
                    % X step, proximal gradient method to 
                    % solve f + g + quadratic term
                    for i=1:100
                        grad_f = -H*X + rho * (X - Y + Lambda);
                        grad_map = (X - wthresh(X - eta*grad_f, 's', mu * eta)) / eta;
                        % disp(i+ "-th X step for SOC: "+ temp_F(X) + ", norm: " + norm(grad_map, 'fro'));
                        if norm(grad_map, 'fro') < 1e-8
                            break;
                        end
                        X = X - eta * grad_map;
                    end
                    
%                     % still X step, try subgradient step
%                     for i=1:100
%                         subg = -H*X / rho + mu/rho * sign(X) + rho * (X - Y + Lambda);
%                         disp(i+ "-th X step for SOC: "+ temp_F(X) + ", norm: " + norm(subg, 'fro'));
%                         if norm(subg, 'fro') < 1e-8
%                             break;
%                         end
%                         X = X - eta * subg;
%                     end
                    
                    % Y step: a projection step
                    [U,~,V] = svd(X + Lambda);
                    Y = U*eye(n,p)*V.';

                    % Lambda step
                    Lambda = Lambda + (X - Y);

                    elapsed_time = toc(admm_start);

                    % Value update
                    F_val_soc(iter) = F(Y);
                    F_val_soc_avg(iter) = F_val_soc_avg(iter) + F(Y);
                    vio_soc_avg(iter) = vio_soc_avg(iter) + norm(Y - X, 'fro');
%                     if abs(F_val_soc(iter) - F_val_soc(iter-1)) <= 1e-8
%                         break
%                     end

                    cpu_time_soc(k,iter) = cpu_time_soc(k,iter) + elapsed_time;
                    if iter < 1000
                        cpu_time_soc(k,iter+1) = cpu_time_soc(k,iter);
                    end

                    % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
                end
                iter0 = min(iter, iter0);
                sparse_soc(k) = sum(sum(abs(Y) <= 1e-8))/(n*p);
                error_soc(k) = norm(Y - X, 'fro');
                
                %% MADMM
                X = X0; Y = X0;
                Lambda = zeros(size(X));
                eta = 1e-2; N1 = 500; rho = 100;
                
                for iter=2:N1
                    admm_start = tic;
                    % X step: a Riemannian gradient step
                    for i=1:100
                        gx = -H*X + rho*(X - Y + Lambda);
                        rgx = proj(X, gx);
                        if norm(rgx, 'fro') < 1e-8
                            break;
                        end
                        X = retr(X, -eta*rgx);
                        %fprintf('inner iter: %d, X step subgrad:%f\n', i, norm(rgx));
                    end

                    % Y step
                    Y = wthresh(X + Lambda ,'s', mu/rho);

                    % Lambda step
                    Lambda = Lambda + (X - Y);

                    elapsed_time = toc(admm_start);

                    % Value update
                    F_val_madmm(iter) = F(X);
                    F_val_madmm_avg(iter) = F_val_madmm_avg(iter) + F(X);
                    vio_madmm_avg(iter) = vio_madmm_avg(iter) + norm(Y - X, 'fro');
%                     if abs(F_val_madmm(iter) - F_val_madmm(iter-1)) <= 1e-8
%                         break
%                     end

                    cpu_time_madmm(k,iter) = cpu_time_madmm(k,iter) + elapsed_time;
                    if iter < 1000
                        cpu_time_madmm(k,iter+1) = cpu_time_madmm(k,iter);
                    end

                    % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
                end
                iter1 = min(iter, iter1);
                sparse_madmm(k) = sum(sum(abs(Y) <= 1e-8))/(n*p);
                error_madmm(k) = norm(Y - X, 'fro');
                

                %% RADMM
                X = X0; Z = X0;
                Lambda = zeros(size(X)); 
                eta = 1e-2; gamma = 1e-9; N2 = 500; rho = 100;
                
                for iter=2:N2
                    admm_start = tic;
                    % X step: a gradient step
                    for i=1:1
                        gx = -H*X + Lambda + rho*(X - Z);
                        rgx = proj(X, gx);
                        X = retr(X, -(eta)*rgx);
                        %fprintf('inner iter: %d, X step subgrad:%f\n', i, norm(rgx));
                    end

                    % Z step (also update Y)
                    Y = wthresh(X + Lambda/rho,'s',mu*(1+rho*gamma)/rho);
                    Z = (Y/gamma + Lambda + rho*X) / (1/gamma + rho);

                    % Lambda step
                    Lambda = Lambda + rho*(X - Z);

                    elapsed_time = toc(admm_start);

                    % Value update
                    F_val_radmm(iter) = F(X);
                    F_val_radmm_avg(iter) = F_val_radmm_avg(iter) + F(X);
                    vio_radmm_avg(iter) = vio_radmm_avg(iter) + norm(Y - X, 'fro');
%                     if abs(F_val_radmm(iter) - F_val_radmm(iter-1)) <= 1e-8
%                         break
%                     end

                    cpu_time_radmm(k,iter) = cpu_time_radmm(k,iter) + elapsed_time;
                    if iter < 1000
                        cpu_time_radmm(k,iter+1) = cpu_time_radmm(k,iter);
                    end

                    % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
                end
                iter2 = min(iter, iter2);
                sparse_radmm(k) = sum(sum(abs(X) <= 1e-8))/(n*p);
                error_radmm(k) = norm(Y - X, 'fro');
                
            end
            F_val_soc_avg = (F_val_soc_avg/avg);
            F_val_madmm_avg = (F_val_madmm_avg/avg);
            F_val_radmm_avg = (F_val_radmm_avg/avg);
            
            vio_soc_avg = (vio_soc_avg/avg);
            vio_madmm_avg = (vio_madmm_avg/avg);
            vio_radmm_avg = (vio_radmm_avg/avg);
            
            cpu_time_soc = sum(cpu_time_soc,1)/avg;
            cpu_time_madmm = sum(cpu_time_madmm,1)/avg;
            cpu_time_radmm = sum(cpu_time_radmm,1)/avg;
            
            av_sparse_soc = sum(sparse_soc)/avg;
            av_sparse_madmm = sum(sparse_madmm)/avg;
            av_sparse_radmm = sum(sparse_radmm)/avg;
            
            av_error_soc = sum(error_soc)/avg;
            av_error_madmm = sum(error_madmm)/avg;
            av_error_radmm = sum(error_radmm)/avg;

            disp("sparisty for SOC, MADMM and RADMM: ")

            disp([av_sparse_soc, av_sparse_madmm, av_sparse_radmm])
            
            disp("error of SOC, MADMM and RADMM: ")
            
            disp([av_error_soc, av_error_madmm, av_error_radmm])
            
            disp("CPU time for SOC, MADMM and RADMM: ")
            
            disp([cpu_time_soc(iter0 - 1), cpu_time_madmm(iter1 - 1), cpu_time_radmm(iter2 - 1)]);
            
            disp("function value for output SOC, MADMM and RADMM: ")

            disp([F_val_soc_avg(iter0 - 1), F_val_madmm_avg(iter1 - 1), F_val_radmm_avg(iter2 - 1)]);

            %% Plots
            figure0 = figure(1);
            clf
            plot(F_val_soc_avg(1:iter0), '-.'); hold on;
            plot(F_val_madmm_avg(1:iter1), '-.'); hold on;
            plot(F_val_radmm_avg(1:iter2)); hold on;
            xlabel("Iterations");
            ylabel("Function value");
            legend('SOC', 'MADMM', 'RADMM');
            legend('Location','best');
            filename = "soc_madmm_n_" + n + "_p_" + p + "_mu_" + mu + "_time_fval.pdf";
            saveas(figure0, filename);
            % figure0.show()
            
            figure1 = figure(2);
            clf
            semilogx(cpu_time_soc(1:iter0), F_val_soc_avg(1:iter0), '-.'); hold on;
            semilogx(cpu_time_madmm(1:iter1), F_val_madmm_avg(1:iter1), '-.'); hold on;
            semilogx(cpu_time_radmm(1:iter2), F_val_radmm_avg(1:iter2)); hold on;
            xlabel("CPU time");
            ylabel("Function value");
            legend('SOC', 'MADMM', 'RADMM');
            legend('Location','best');
            filename = "soc_madmm_cpu_time_n_" + n + "_p_" + p + "_mu_" + mu + "_time_fval.pdf";
            saveas(figure1, filename);
            % figure1.show()
            
            figure2 = figure(3);
            clf
            semilogx(cpu_time_soc(1:iter0), vio_soc_avg(1:iter0), '-.'); hold on;
            semilogx(cpu_time_madmm(1:iter1), vio_madmm_avg(1:iter1), '-.'); hold on;
            semilogx(cpu_time_radmm(1:iter2), vio_radmm_avg(1:iter2)); hold on;
            xlabel("CPU time");
            ylabel("Constrain violation");
            legend('SOC', 'MADMM', 'RADMM');
            legend('Location','best');
            % filename = "soc_madmm_n_" + n + "_p_" + p + "_mu_" + mu + "_time_fval.pdf";
            % saveas(figure1, filename);
            % figure1.show()
        end
    end
end

