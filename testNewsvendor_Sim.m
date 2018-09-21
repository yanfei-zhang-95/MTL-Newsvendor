clc
clear all
rng('default')
rng(0)
%% Prepare  data
% The max iterations
maxiter=1000;

% Type of loss function
type='Quad';  %'Quad';  %Eucl
scale = 1;  % or 0 

%cp and ch
ch=1.5;
cp=3;

load NewsVendorData  %Xtrain Dtrain XtrainB Xtest Dtest XtestB %Original

D = size(XtrainB,1); % The number of input features (43 in example 2 of the paper)
                     % This is binary feature for categorical data
n_hid1 = 282;   % The number of neurons on the first hidden layer
                % The paper use 350
n_hid2 = 60;   % The number of neurons on the second hidden layer
                % The paper uses 100
                % You can change both n_hid1 and n_hid2 to other values                             

N = size(XtrainB,2);         % The total of number of training data
maxD = max(Dtrain);
minD = min(Dtrain);

if scale
    demands = (Dtrain - minD)/(maxD -  minD);
else
    demands = Dtrain;   % not scale 
end

% Define the neural network structure
f=1/10;   %1/66 ;       % To define the scale for random weights

net.layers = {} ;
% We use a linear mapping from input to the first hidden layer then apply 
% sigmoid activation
W1 = f*randn(n_hid1, D);
B1 = zeros(n_hid1,1);
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{W1, B1}}) ;   % First hidden layer
net.layers{end+1} = struct('type','sigmoid');

% We also use a linear mapping from the first to the second hidden layer
% and apply sigmoid
W2 = f*randn(n_hid2, n_hid1);
B2 = zeros(n_hid2,1);
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{W2, B2}}) ;
net.layers{end+1} = struct('type','sigmoid');


% Output to a single estimate for demand, note we increase scale 10 times
Wo = 60*randn(1,n_hid2);
Bo = mean(demands);  %zeros(1,1);

net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{Wo, Bo}}) ;
 
% Here we test the cost function used in the paper
switch type
    case 'Quad'
        net.layers{end+1} = struct('type', 'newsvendorloss_l2', ...
                            'ch', ch, 'cp', cp, 'demands', demands) ;
    case 'Eucl'          
        net.layers{end+1} = struct('type', 'newsvendorloss', ...
                            'ch', ch, 'cp', cp, 'demands', demands) ;
end
% lambda for weights regularization
lambda = 0.01;
% Beta is not used
beta = 0;
%load Predicted_Q
theta = [W1(:); B1(:); W2(:); B2(:); Wo(:); Bo(:)];
%theta = opttheta_Q;

% The following line was used to test the correctness of the code.
%[cost,grad] = newsvendorCost(theta, net, XtrainB, lambda, beta);
%%

if(isdeployed==false) % This is necessary if it is deployed. No addpath in deploy version. 
    % You need to change to your path to minFunc.
    addpath('./minFunc/')
    % addpath('/Users/jbgao/Documents/Gaofiles/matlab/Hinton/UFLDL/starter/minFunc/')
end
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
% function value and the gradient. In our problem,
% MatrixNeuralNetworkRegressionCost.m satisfies this.
options.maxIter = maxiter;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.MaxFunEvals = 2000;	

%[opttheta, costTr, exitflag, info] = minFunc( @(p) newsvendorCost(p, net, XtrainB, lambda, beta), ini_weight, options);
[opttheta, costTr, exitflag, info] = minFunc( @(p) newsvendorCost(p, net, XtrainB, lambda, beta), theta, options);

if(isdeployed==false)
    rmpath('./minFunc/')
    %rmpath('/Users/jbgao/Documents/Gaofiles/matlab/Hinton/UFLDL/starter/minFunc/')
end
%% Prediction on test set
L = numel(net.layers);
count_end = 0;
for i_ = 1:L
    if isfield(net.layers{i_}, 'weights')
        m = numel(net.layers{i_}.weights);
        for j_=1:m
            count_start = count_end + 1;
            w_size = size(net.layers{i_}.weights{j_});
            count_end = count_start + prod(w_size) - 1;
            net.layers{i_}.weights{j_} = reshape(opttheta(count_start:count_end), w_size); 
        end
    end
end
res_Te = SimpleNN(net, XtestB, [], [], 'prediction', 1, 'beta', beta);
predicted_demand_Te = res_Te(L).x;

res_Tr=SimpleNN(net, XtrainB, [], [], 'prediction', 1, 'beta', beta);
predicted_demand_Tr=res_Tr(L).x;
%%
%n=numel(Dtest);
if scale
    predicted_demand_Te = minD + (maxD - minD)*predicted_demand_Te;
    predicted_demand_Tr = minD + (maxD - minD)*predicted_demand_Tr; 
end

test_err = mean((Dtest-predicted_demand_Te).^2);
test_errors= Dtest - predicted_demand_Te;
costTe=mean(cp * max(Dtest - predicted_demand_Te, 0.0) + ch * max ( predicted_demand_Te - Dtest, 0.0));

 
train_err = mean((Dtrain-predicted_demand_Tr).^2);
train_errors = Dtrain -predicted_demand_Tr;
costTr= mean(cp * max(Dtrain - predicted_demand_Tr, 0.0) + ch * max ( predicted_demand_Tr - Dtrain, 0.0));



switch type
    case 'Quad'
        predicted_Te_Q=predicted_demand_Te;
        predicted_Tr_Q=predicted_demand_Tr;
        test_err_Q=test_err;
        train_err_Q = train_err;
        test_errors_Q=test_errors;
        train_errors_Q = train_errors;
        costTe_Q=costTe;
        costTr_Q = costTr;
        net_Q = net;
        opttheta_Q = opttheta;
        save('predicted_Q','net_Q','test_err_Q','train_err_Q', 'costTe_Q','costTr_Q', 'test_errors_Q', 'train_errors_Q', 'opttheta_Q');
    case 'Eucl'
        predicted_Te_E=predicted_demand_Te;
        predicted_Tr_E=predicted_demand_Tr;
        test_err_E=test_err;
        train_err_E = train_err;
        test_errors_E=test_errors;
        train_errors_E = train_errors;
        costTe_E=costTe;
        costTr_E = costTr;
        net_E = net;
        opttheta_E = opttheta;
        save('predicted_E','net_E','test_err_E','train_err_E', 'costTe_E','costTr_E', 'test_errors_E', 'train_errors_E', 'opttheta_E');
end

 
 