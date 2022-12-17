function r = retr_stiefel(V,Y)
    % retraction to Stiefel manifold at Y
%     [n, p] = size(X);
%     [U,~,V] = svd(X+Y);
%     r = U*eye(n,p)*V.';
    
%     [U, SIGMA, S] = svd((V+Y)'*(V+Y));   
%     SIGMA =diag(SIGMA);    
%     r = (V+Y)*(U*diag(sqrt(1./SIGMA))*S');
    [~, p] = size(Y);
    [Q, ~] = qr(V+Y);
    r = Q(:,1:p);
end