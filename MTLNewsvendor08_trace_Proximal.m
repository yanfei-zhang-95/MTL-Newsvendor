clc;
clear all;
close all;

rand('seed', 1e6)
randn('seed', 1e6)
%% Data Preparation
load NewsvendorData_Label;
[~,train_]=size(Dtrain);

%% Compulsory Parameters
n_hid1 = 10; %First hidden layer
n_hid2 = 100; %Second hidden layer
D = size(XtrainB, 1);%Number of input features
[~, O] = size(unique(Ltrain));

%demands = Dtrain;
minDtrain = min(Dtrain);
maxDtrain = max(Dtrain);

%% Junbin's comments   8 September 2017
% I said many times it is better to normalize your data when training
% neural networks. 
demands = Dtrain;

index = zeros(O, train_);
count_start = 1;

for i = 1:O
    label = length(find(Ltrain == i));
    count_end = count_start + label;
    index(i, count_start:count_end) = ones(1, count_end - count_start + 1);
    count_start = count_end;
end
index = index(:, 1:end-1);
%ch = rand(1, O) + 1;
%cp = 3 * ch;
 
 
% I still find it seems to me you need sharpen your programming skills too.
%% Building the network   Junbin Gao
net.layers={} ;
f=1/10 ; 
%% Define a network
% First hidden layer
W1 = f*randn(n_hid1, D);
B1 = zeros(n_hid1,1);
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{W1, B1}}) ;
net.layers{end+1} = struct('type','sigmoid');

% Second hidden layer
W2 = f*randn(n_hid2, n_hid1);
B2 = zeros(n_hid2,1);
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{W2, B2}}) ;
net.layers{end+1} = struct('type','sigmoid');

% Third hidden layer 
Wo = 6*randn(O,n_hid2);
Bo = zeros(O,1);
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{Wo, Bo}}) ;                 
                       
% lambda for weights regularization
lambda = 0.1;
gamma = 0.1;

% Beta is not used
beta = 0;

theta = [W1(:); B1(:); W2(:); B2(:); Wo(:); Bo(:)];

ch = rand(1, O) + 1;
cp = 3 * ch;
  
net.layers{end+1} = struct('type', 'newsvendorloss_MTL', ...
                            'ch', ch, 'cp', cp, 'demands', demands , 'index', index) ;

%%
[TrainErr ,grad] = newsvendorCost_trace(theta, net, XtrainB, lambda,gamma, beta);

%% Junbin's comments   8 September 2017
% On your code, always check this test first. If this error is not at a
% scale of 10^-9 then something wrong with BP algorithm.
% func = @(alpha) newsvendorCost_l1(alpha, net, XtrainB, lambda, beta ) ; 
% err = tnncheckDerivativeNumerically(func, theta, grad) ;
% disp(err)
%%
L = numel(net.layers);
count_end = 0;
grad_ = cell(1,6);
w_ = cell(1,6);
for i = 1:L
    if isfield(net.layers{i}, 'weights')
        m = length(net.layers{i}.weights);
        for j=1:m
            count_start = count_end + 1;
            w_size = size(net.layers{i}.weights{j});
            count_end = count_start + prod(w_size) - 1;
            w_{i+j-1} = reshape(theta(count_start:count_end), w_size); 
            grad_{i+j-1} = reshape(grad(count_start:count_end), w_size); 
        end
    end    
end
%%
maxIter = 1000;
alpha = 0.01; % learning rate for other parameters
% Lip = max(ch); % Lipschitz constant
fval_ = zeros(1, maxIter);
for i = 1:maxIter
    % Gradient Descent for the Rest of Parameters
    for j = 1:L-2
        w_{j} = w_{j} - alpha*grad_{j};
    end
    w_{6} = w_{6} - alpha*grad_{6};
    
    % First Update
    theta = [w_{1}(:); w_{2}(:); w_{3}(:); w_{4}(:); w_{5}(:); w_{6}(:)];
    [~, grad] = newsvendorCost_trace(theta, net, XtrainB, lambda, gamma, beta);
    count_end = 0;
    grad_ = cell(1,6);
    w_ = cell(1,6);
    for p = 1:L
        if isfield(net.layers{p}, 'weights')
            m = length(net.layers{p}.weights);
            for q=1:m
                count_start = count_end + 1;
                w_size = size(net.layers{p}.weights{q});
                count_end = count_start + prod(w_size) - 1;
                w_{p+q-1} = reshape(theta(count_start:count_end), w_size); 
                grad_{p+q-1} = reshape(grad(count_start:count_end), w_size); 
            end
        end    
    end
    
    % Proximal Algorithm
    Lip = norm(grad_{5},inf);
    for k = 1:5
        [U, S, V] = svd(grad_{5});
        diag_ = diag(S);
        a = 0;
        for r = 1:numel(diag_)
            if diag_(r)>gamma/Lip
                a = a + 1;
            end
        end
        [U_, S_, V_] = svds(grad_{5}, a);
         w_{5} = U_* (S_ - gamma/Lip * eye(a)) * V_';
    end    
    
    % Second Update
    theta = [w_{1}(:); w_{2}(:); w_{3}(:); w_{4}(:); w_{5}(:); w_{6}(:)];
    [fval_2, grad] = newsvendorCost_trace(theta, net, XtrainB, lambda, gamma, beta);
    count_end = 0;
    grad_ = cell(1,6);
    w_ = cell(1,6);
    for p = 1:L
        if isfield(net.layers{p}, 'weights')
            m = length(net.layers{p}.weights);
            for q=1:m
                count_start = count_end + 1;
                w_size = size(net.layers{p}.weights{q});
                count_end = count_start + prod(w_size) - 1;
                w_{p+q-1} = reshape(theta(count_start:count_end), w_size); 
                grad_{p+q-1} = reshape(grad(count_start:count_end), w_size); 
            end
        end    
    end
    
    % Save Results
    fval_(i) = fval_2;
    disp(i);   
    % Setting Conditions 
%     if i>1
%         if fval_(i) - fval_(i-1)> 0
%             fval_(fval_ == 0) = [];
%             disp('Convergence Reached');
%             break;
%         end
%     end
    disp(fval_2);
    disp(a);
end
opttheta = theta;


%% optimal training error
[optTrainErr,~] = newsvendorCost_l1(opttheta, net, XtrainB, lambda, gamma, beta);
%% Prediction
[~,test_]=size(Dtest);
res = SimpleNN(net, XtestB, [], [], 'prediction', 1, 'beta', beta);
predicted_demand = res(L).x;
predicted_demand_modified = zeros(1, test_);

% There is a shifting in your code
% count_start = 1;
count_start = 0;
% I revised the following code
for i = 1:O
    label = length(find(Ltest == i));
    count_end = count_start + label;
    predicted_demand_modified(count_start+1:count_end) = predicted_demand(i, count_start+1:count_end);      
    count_start = count_end;
end
TestErr = (Dtest - predicted_demand_modified).^2;
TestErr = mean(TestErr);
disp(TestErr);
%%
figure;
imagesc(net.layers{5}.weights{1}); colorbar;
figure;
plot(fval_);