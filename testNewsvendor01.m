clear all
rng('default')
rng(0)

% Prepare  data
D = csvread('Basket_train_data_binary.csv');
XtrainB = D(:,1:43)';
Dtrain = D(:,44)';
D = csvread('Basket_train_data.csv');
Xtrain = D(:,1:3)';
D = csvread('Basket_test_data_binary.csv');
XtestB = D(:,1:43)';
Dtest = D(:,44)';
D = csvread('Basket_test_data.csv');
Xtest = D(:,1:3)';
save NewsVendorData  Xtrain Dtrain XtrainB Xtest Dtest XtestB


D = 20;   % The number of input features (43 in example 2 of the paper)
n_hid1 = 16;    % The number of neurons on the first hidden layer
n_hid2 = 10;    % The number of neurons on the second hidden layer
                % You can change both n_hid1 and n_hid2 to other values

N = 50;         % The total of number of training data
 
x  = randn(D,N);  % The data
demands = rand(1,N);

% Define the neural network structure
f=1/100 ;       % To define the scale for random weights

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
Wo = 10*f*randn(1,n_hid2);
Bo = zeros(1,1);
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{Wo, Bo}}) ;

% Finally we define the cost layer
net.layers{end+1} = struct('type', 'newsvendorloss', ...
                            'ch', 2, 'cp', 1, 'demands', demands) ;

% lambda for weights regularization
lambda = 0.01;

% Beta is not used
beta = 0;

theta = [W1(:); B1(:); W2(:); B2(:); Wo(:); Bo(:)];

% The following line was used to test the correctness of the code.
[cost,grad] = newsvendorCost(theta, net, x, lambda, beta);

% I have confirmed the derivative formula is correct by the following code
%
% func = @(alpha) newsvendorCost(alpha, net, x, lambda, beta ) ; 
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
options.maxIter = 100;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.MaxFunEvals = 2000;	

[opttheta, cost, exitflag, info] = minFunc( @(p) newsvendorCost(p, net, x, lambda, beta), theta, options);

if(isdeployed==false)
    rmpath('./minFunc/')
    %rmpath('/Users/jbgao/Documents/Gaofiles/matlab/Hinton/UFLDL/starter/minFunc/')
end
