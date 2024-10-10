function [X_radmm, F_radmm,sparsity_radmm,time_radmm,error_XZ, iter_radmm, flag_succ]=radmm_spca(B,option)
%min -Tr(X'*A*X)+ mu*norm(X,1) s.t. X'*X=Ir.
% A = B'*B type = 0 or A = B  type = 1
tic;
r = option.r;
n = option.n;
mu = option.mu;
maxiter =option.maxiter;
tol = option.tol;
type = option.type;
if type==0 % data matrix
    A = -B'*B;
else
    A = -B;
end

h=@(X) mu*sum(sum(abs(X)));
%rho = svds(B,1)^2 + r/2;%  stepsize
%rho = svds(B,1)^2 + n*r*mu/25 + 1;
%rho = svds(B,1)^2 + n/50 ;% good for mu and r
%rho = 2* svds(B,1)^2  ;%  n/30 not converge   1.9* sometimes not converge
rho = 1e2;
lambda = rho;
gamma = 1e-6;
eta = 1e-2;

% init all points
% P = option.phi_init;    Q = P;
% Z = zeros(n,r); 
% b=Z;
X = option.phi_init;
Z = X; % no need to initialized Y
Lambda = zeros(n,r);

F_ad=zeros(maxiter,1);
not_full_rank = 0;

%chA = chol( 2*A + (r+lambda)*eye(d));
% Ainv = inv( 2*A + (rho+lambda)*eye(n));
flag_maxiter = 0;

for itera=1:maxiter
    % X step: a gradient step
    for i=1:1
        gx = A*X + Lambda + rho*(X - Z);
        rgx = proj(X, gx);
        X = retr(X, -(eta)*rgx);
        %fprintf('inner iter: %d, X step subgrad:%f\n', i, norm(rgx));
    end

    % Z step (also update Y)
    Y = wthresh(X+Lambda/rho,'s',mu*(1+rho*gamma)/rho);
    Z = (Y/gamma + Lambda + rho*X) / (1/gamma + rho);

    % Lambda step
    Lambda = Lambda + rho*(X - Z);
    
    if itera>2
        normXZ = norm(X-Z,'fro');
        normZ = norm(Z,'fro');
        normX = norm(X,'fro');
        normLam = norm(Lambda,'fro');
        if  normXZ/max(1,max(normZ,normX)) <tol
            if type == 0 % data matrix
                AX = -(B'*(B*X));
            else
                AX = -(B*X);
            end
            F_ad(itera) = sum(sum(X.*(AX))) / 2+h(X);
            % if F_ad(itera)<= option.F_manpg + 1e-7
            %     break;
            % end
            if itera > 1 && abs(F_ad(itera) - F_ad(itera-1)) <= 1e-8
                break
            end
        end
        %         if   normXQ  + normXP <1e-9*r
        %             break;
        %         end
    end
    
    X_old=X;
    if itera ==maxiter
        flag_maxiter =1;
    end
end
% X((abs(X)<=1e-5))=0;
X_radmm=X;
time_radmm= toc;
error_XZ = norm(X-Z,'fro');
% X_manpg = option.X_manpg;
sparsity_radmm= sum(sum(abs(X)<=1e-5))/(n*r);
if itera == maxiter
    flag_succ = 0; %fail
    F_radmm = F_ad(itera);
    iter_radmm = itera;
    fprintf('RADMM does not converge to stationary\n');
    
    fprintf('RADMM:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
    
    print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
    fprintf(1,print_format, itera, F_ad(itera), time_radmm, sparsity_radmm,  error_XZ);
    % time_radmm = 0;
else
    % if norm(X_manpg*X_manpg'- X_radmm*X_radmm','fro')^2 > 0.1
    %     fprintf('RADMM returns different point \n');
    %     fprintf('RADMM:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
    % 
    %     print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
    %     fprintf(1,print_format, itera, F_ad(itera), time_radmm, sparsity_radmm,  error_XZ);
    %     flag_succ = 2; % different point
    %     F_radmm = 0;
    %     sparsity_radmm = 0;
    %     iter_radmm = 0;
    % 
    %     time_radmm = 0;
    % else
        
    flag_succ = 1; % success
    F_radmm = F_ad(itera);
    iter_radmm = itera;
    % residual_Q = norm(Q'*Q-eye(n),'fro')^2;
    fprintf('RADMM:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
    
    print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
    fprintf(1,print_format, itera, F_ad(itera), time_radmm, sparsity_radmm,  error_XZ);
    % end
end
