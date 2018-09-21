clc;
clear;
close;
%% Data Preparation
load NewsvendorData;
[~,train_]=size(Dtrain);
TestErr_=0;
for k=1:500
    %% Compulsory Parameters
    n_hid1=7; %First hidden layer
    n_hid2=8; %Second hidden layer
    D=size(XtrainB,1);%Number of input features
    O=2; %Number of outputs
    demands=Dtrain/max(Dtrain);
    index=[ones(1,ceil(train_/O)) zeros(1,floor(train_/O)); zeros(1,ceil(train_/O)) ones(1,floor(train_/O))];
    ch=[1,1.5];
    cp=3*ch;
    %% Building the network   Junbin Gao
    net.layers={} ;
    f=1/130; 
    %% Define a network
    % First hidden layer
    W1 = f*randn(n_hid1, D);
    B1 = zeros(n_hid1,1);
    net.layers{end+1} = struct('type', 'linear', ...
                               'weights', {{W1, B1}}) ;
    net.layers{end+1} = struct('type','relu');
    % Second hidden layer
    W2 = f*randn(n_hid2, n_hid1);
    B2 = zeros(n_hid2,1);
    net.layers{end+1} = struct('type', 'linear', ...
                               'weights', {{W2, B2}}) ;
    net.layers{end+1} = struct('type','relu');
    % Third hidden layer 
    Wo = 60*f*randn(O,n_hid2);
    Bo = zeros(O,1);
    net.layers{end+1} = struct('type', 'linear', ...
                               'weights', {{Wo, Bo}}) ;
    delta=[0.2,0.8];
    % lambda for weights regularization
    lambda = 1/138;
    beta = 0;
    theta = [W1(:); B1(:); W2(:); B2(:); Wo(:); Bo(:)];

    %% Junbin's comments   15 August 2017
    % Now in the last layer (i.e., newsvendorloss_MTL) we all need store the index
    % information.  You may add 'index', index  after demands in the following
    % statement
    % Thus we need to change the calculate in newsvendorCost.m function for
    % this new MTL loss layer
    net.layers{end+1} = struct('type', 'newsvendorloss_MTL', ...
                                'ch', ch, 'cp', cp, 'demands', demands , 'index', index) ;
    %%
    [TrainErr,grad] = newsvendorCost_new(theta, net, XtrainB, lambda, beta, delta);
    %%
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
    options.maxIter = 2000;	  % Maximum number of iterations of L-BFGS to run
    options.display = 'off';
    options.LS=0;
    options.MaxFunEvals = 2000;	

    [opttheta, cost, exitflag, info] = minFunc( @(p) newsvendorCost_new(p, net, XtrainB, lambda, beta, delta), theta, options);

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

    %% TestError
    [~,test_]=size(Dtest);
    demands=Dtest/max(Dtest);
    index=[ones(1,ceil(test_/O)) zeros(1,floor(test_/O)); zeros(1,ceil(test_/O)) ones(1,floor(test_/O))];
    net.layers{end} = struct('type', 'newsvendorloss_MTL', ...
                                'ch', ch, 'cp', cp, 'demands', demands , 'index', index) ;
    [TestErr,~] = newsvendorCost_new(opttheta, net, XtestB, lambda, beta, delta);
    TestErr_(end+1)=TestErr;
    disp(k);
end
TestErr_=TestErr_(2:end);
plot(TestErr_);
min_err=min(TestErr_)
min_err_=find(TestErr_==min_err)