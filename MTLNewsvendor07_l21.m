clc;
clear all;
close all;

rand('seed', 1e6)
randn('seed', 1e6)


%% Data Preparation
load NewsvendorData_Label;
[~,train_]=size(Dtrain);

%% Compulsory Paramete  rs
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
% demands = (Dtrain - minDtrain)/(maxDtrain-minDtrain);
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


for i = [3, 5, 8, 9 10]
    index(i,:) = zeros(1,train_);
end
 
 
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
%% Junbin's comments   8 September 2017
% I found a big problem here where you used relu activation instead of
% sigmoid. As there are so many zeros in feature, this will make you apply
% relu on zero values where relu is not differentiable. That is why the
% derivative calculation was wrong.

% Second hidden layer
W2 = f*randn(n_hid2, n_hid1);
B2 = zeros(n_hid2,1);
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{W2, B2}}) ;
net.layers{end+1} = struct('type','sigmoid');

% Third hidden layer 
Wo = 60*f*randn(O,n_hid2);
Bo = zeros(O,1);
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{Wo, Bo}}) ;                 
                       
% lambda for weights regularization
lambda = 0.01;   %0.001;

% Beta is not used
beta = 0;

theta = [W1(:); B1(:); W2(:); B2(:); Wo(:); Bo(:)];


%% Junbin's comments   8 September 2017
% Now you may try other random ch and cp
ch = rand(1, O) + 1;
cp = 3 * ch;
  
net.layers{end+1} = struct('type', 'newsvendorloss_MTL', ...
                            'ch', ch, 'cp', cp, 'demands', demands , 'index', index) ;

%%
[TrainErr,grad] = newsvendorCost_l21(theta, net, XtrainB, lambda, beta);

%%
% func = @(alpha) newsvendorCost_l21(alpha, net, XtrainB, lambda, beta) ; 
% err = tnncheckDerivativeNumerically(func, theta, grad) ;
% disp(err)
%% Junbin's comments   8 September 2017
% On your code, always check this test first. If this error is not at a
% scale of 10^-9 then something wrong with BP algorithm.


if(isdeployed==false) % This is necessary if it is deployed. No addpath in deploy version. 
    % You need to change to your path to minFunc.
    addpath('./minFunc/')
    % addpath('/Users/jbgao/Documents/Gaofiles/matlab/Hinton/UFLDL/starter/minFunc/')
end
options.Method = 'lbfgs'; 
options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.MaxFunEvals = 1000;	

[opttheta, cost, exitflag, info] = minFunc( @(p) newsvendorCost_l21(p, net, XtrainB, lambda, beta), theta, options);

if(isdeployed==false)
    rmpath('./minFunc/')
    %rmpath('/Users/jbgao/Documents/Gaofiles/matlab/Hinton/UFLDL/starter/minFunc/')
end
%%
L = numel(net.layers);
count_end = 0;
for i = 1:L
    if isfield(net.layers{i}, 'weights')
        m = length(net.layers{i}.weights);
        for j=1:m
            count_start = count_end + 1;
            w_size = size(net.layers{i}.weights{j});
            count_end = count_start + prod(w_size) - 1;
            net.layers{i}.weights{j} = reshape(opttheta(count_start:count_end), w_size); 
        end
    end    
end
%% optimal training error
[optTrainErr,~] = newsvendorCost_l21(opttheta, net, XtrainB, lambda, beta);
%% Prediction
[~,test_]=size(Dtest);
res = SimpleNN(net, XtestB, [], [], 'prediction', 1, 'beta', beta);
predicted_demand = res(L).x;
predicted_demand_modified = zeros(1, test_);

count_start = 0;
index = ones(1, test_);
for i = 1:O
    label = length(find(Ltest == i));
    count_end = count_start + label;
    predicted_demand_modified(count_start+1:count_end) = predicted_demand(i, count_start+1:count_end);
    if i == 3 || i == 5 || i == 8 || i == 9 || i == 10
         index(count_start+1:count_end) = zeros(1, count_end-count_start);
        predicted_demand_modified(count_start+1:count_end) = zeros(1, count_end-count_start);
    end
    count_start = count_end;
end

TestErr = index .* (Dtest - predicted_demand_modified).^2;
TestErr_ = 0;
for i = 1:test_
    if TestErr(i) ~= 0
        TestErr_(end+1) = TestErr(i);
    end
end
TestErr_ = TestErr_(2:end);
TestErr = mean(TestErr_);
disp(TestErr);
imagesc(net.layers{5}.weights{1}); colorbar;