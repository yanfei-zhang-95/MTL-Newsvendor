clc;
clear all
rng('default')
rng(0)
%%
% Prepare  data
% Data was converted matlab format
load NewsVendorData_Label  %Ltrain Dtrain XtrainB Ltest Dtest XtestB

minDtrain = min(Dtrain);
maxDtrain = max(Dtrain);

%%
% Training data are XtrainB (binary features), Xtrain (no use) and Dtrain
% (demands)
% Testing data are XtestB (binary features), Xtest (no use) and Dtest
% (demands)
% After networking training, we need test the network on test data

D = size(XtrainB,1); % The number of input features (43 in example 2 of the paper)
                     % This is binary feature for categorical data
n_hid1 = 10;   % The number of neurons on the first hidden layer
                % The paper use 350
n_hid2 = 10;   % The number of neurons on the second hidden layer
                % The paper uses 100
                % You can change both n_hid1 and n_hid2 to other values
N = size(XtrainB,2);         % The total of number of training data

% demands = Dtrain;  % We scale it for better training
demands = (Dtrain - minDtrain)/(maxDtrain-minDtrain);

% Define the neural network structure
f=1/10 ;       % To define the scale for random weights

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
Wo = 60*f*randn(1,n_hid2);
Bo = 100;
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{Wo, Bo}}) ;
 
% Finally we define the cost layer
net.layers{end+1} = struct('type', 'newsvendorloss', ...
                            'ch', 1, 'cp', 3, 'demands', demands) ;

% lambda for weights regularization
lambda = 0.01;

% Beta is not used
beta = 0;

theta = [W1(:); B1(:); W2(:); B2(:); Wo(:); Bo(:)];

% The following line was used to test the correctness of the code.
[TrainErr,grad] = newsvendorCost(theta, net, XtrainB, lambda, beta);
%%
% I have confirmed the derivative formula is correct by the following code
%
% func = @(alpha) newsvendorCost(alpha, net, XtrainB, lambda, beta ) ; 
% err = tnncheckDerivativeNumerically(func, theta, grad) ;
% disp(err)

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
options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.MaxFunEvals = 1000;	

[opttheta, cost, exitflag, info] = minFunc( @(p) newsvendorCost(p, net, XtrainB, lambda, beta), theta, options);

if(isdeployed==false)
    rmpath('./minFunc/')
    %rmpath('/Users/jbgao/Documents/Gaofiles/matlab/Hinton/UFLDL/starter/minFunc/')
end
%%
% I have not written code for prediction

% We collect the best weights from training
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

res = SimpleNN(net, XtestB, [], [], 'prediction', 1, 'beta', beta);

predicted_demand = res(L).x;

n_Dtest =  (Dtest - minDtrain) / (maxDtrain-minDtrain);
TestErr = (n_Dtest - predicted_demand).^2;
TestErr = mean(TestErr);
disp(TestErr);