clc;
clear all;
close all;

rand('seed', 1e6)
randn('seed', 1e6)


%% Data Preparation
load sarcos_inv_test;
mtx = sarcos_inv_test;
XB = mtx(:,1:21);
XtrainB = XB';
D = mtx(:,22:28);
Dtrain = D';

[~,train_]=size(Dtrain);

%% Compulsory Parameters
n_hid1 = 10; %First hidden layer
n_hid2 = 10; %Second hidden layer
D = size(XtrainB, 1);%Number of input features

minDtrain = min(Dtrain(:));
maxDtrain = max(Dtrain(:));

%% Junbin's comments   8 September 2017
demands = (Dtrain - minDtrain)/(maxDtrain-minDtrain);
[O,P] = size(Dtrain);
index = ones(O,P);
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
lambda = 0.0001;   %0.001;

% Beta is not used
beta = 0;

theta = [W1(:); B1(:); W2(:); B2(:); Wo(:); Bo(:)];


%% Junbin's comments   8 September 2017
% Now you may try other random ch and cp
ch = rand(1, O) + 1;
cp = 3 * ch;
  
net.layers{end+1} = struct('type', 'newsvendorloss01', ...
                            'ch', ch, 'cp', cp, 'demands', demands , 'index', index) ;

%%
[TrainErr,grad] = newsvendorCost(theta, net, XtrainB, lambda, beta);

%% Junbin's comments   8 September 2017
% On your code, always check this test first. If this error is not at a
% scale of 10^-9 then something wrong with BP algorithm.
func = @(alpha) newsvendorCost(alpha, net, XtrainB, lambda, beta ) ; 
err = tnncheckDerivativeNumerically(func, theta, grad) ;
disp(err)
%% Optimisation
if(isdeployed==false) % This is necessary if it is deployed. No addpath in deploy version. 
    % You need to change to your path to minFunc.
    addpath('./minFunc/')
    % addpath('/Users/jbgao/Documents/Gaofiles/matlab/Hinton/UFLDL/starter/minFunc/')
end
options.Method = 'lbfgs'; 
options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.MaxFunEvals = 1000;	

[opttheta, cost, exitflag, info] = minFunc( @(p) newsvendorCost(p, net, XtrainB, lambda, beta), theta, options);

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
[optTrainErr,~] = newsvendorCost(opttheta, net, XtrainB, lambda, beta);
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
n_Dtest =  (Dtest - minDtrain) / (maxDtrain-minDtrain);
TestErr = (n_Dtest - predicted_demand_modified).^2;
TestErr = mean(TestErr);
disp(TestErr);