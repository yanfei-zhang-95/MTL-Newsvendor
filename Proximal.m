clc
clear
close
%%
N = 1000;
d = 5;
demand = randn(1,N);
q_0 = randn(d,N);
z = randn(N,N);
x = q_0*z;
index = zeros(d,N);
count_start = 1;
for i = 1:d
    count_end = count_start + 200;
    index(i,count_start:count_end-1)= ones(1,200);
    count_start = count_end;
end
cp=2;
ch=3;
grad = ((-repmat(cp,1,N).* ((repmat(demand,d,1)-x)>0.0)+...
    repmat(ch,1,N).*((repmat(demand,d,1)-x)<0.0)).*index);

%%
gamma = 36;
L = ch;
q_0 = randn(d,N);
z = randn(N,N);
for i = 1:100
    x = q_0*z;
    Y = q_0-1/L*((-repmat(cp,1,N).* ((repmat(demand,d,1)-x)>0.0)+...
    repmat(ch,1,N).*((repmat(demand,d,1)-x)<0.0)).*index);
    [U, S, V] = svd(Y);
    sigma = diag(S);
    k_j = 0;
    for j = 1:numel(sigma)
        if sigma(j)>gamma/L
            k_j = k_j+1;
        end
    end
    [U_0, S_0, V_0] = svds(Y, k_j);
    M = size(S_0);
    q_1 = U_0 * (S_0 - gamma/L*eye(M)) * V_0';
    q_0 = q_1;
    disp(k_j);
end