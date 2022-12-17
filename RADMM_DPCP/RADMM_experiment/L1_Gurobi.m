function b = L1_Gurobi(X,n,pb)

% dimensions and initializations
[D, N] = size(X);
e = ones(N,1);
[~, k ] = size(pb);
%% set up Gurobi model
%clear model
%model.A = sparse([X' -X' eye(N) -eye(N); n' -n' zeros(1,2*N); eye(D) zeros(D) zeros(D,2*N); zeros(D) eye(D) zeros(D,2*N); zeros(N,2*D) eye(N) zeros(N); zeros(N,2*D) zeros(N) eye(N)]);
%model.rhs = [zeros(N,1); 1; zeros(2*D,1); zeros(N,1); zeros(N,1)];

npb = [n' -n'; pb'  -pb'];
npb_rhs = [1;zeros(k,1)];
clear model
model.A = sparse([X' -X' eye(N) -eye(N); npb  zeros(1+k,2*N); eye(D) zeros(D) zeros(D,2*N); zeros(D) eye(D) zeros(D,2*N); zeros(N,2*D) eye(N) zeros(N); zeros(N,2*D) zeros(N) eye(N)]);
model.rhs = [zeros(N,1); npb_rhs; zeros(2*D,1); zeros(N,1); zeros(N,1)];
model.obj = [zeros(1,2*D) e' e'];

% construct the sense string
sense_string = '=';
for i = 1 : N
   sense_string = strcat(sense_string,'='); 
end
for i = 1 : 2*(N+D)+k
    sense_string = strcat(sense_string,'>');
end
model.sense = sense_string;
model.vtype = 'C';

clear params
params.outputflag = 0;
%params.Method = 0;
%params.FeasibilityTol = 1e-5;
%params.OptimalityTol = 1e-5;

% run Gurobi
result = gurobi(model,params);
composite = result.x;
b = composite(1:D) - composite(D+1:2*D);
b = b / norm(b);

end

