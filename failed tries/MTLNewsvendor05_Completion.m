clc;
clear;
close;
%% Data Preparation
load NewsvendorData_Label_;
%% Compulsory Parameters
[~,train_] = size(Dtrain);
n_hid1 = 10; %First hidden layer
n_hid2 = 10; %Second hidden layer
D = size(XtrainB, 1);%Number of input features
[~, O] = size(unique(Ltrain));
%% Index for different products
index = zeros(O, train_);
count_start = 1;
for i = 1:O
    label = length(find(Ltrain == i));
    count_end = count_start + label;
    index(i, count_start:count_end) = ones(1, count_end - count_start + 1);
    count_start = count_end;
end
index = index(:, 1:train_);
%% Randomly set some values to -1
sead = randi([1,train_], 300, 1);
Dtrain(1, sead) = -1;
index_ref = [index; Dtrain];
[rest, ~] = size(index_ref);
rest = rest - 2;
for i = 1:train_
    if index_ref(end, i) == -1
        weights = rand(rest);
        weights_ = sum(weights);
        weights = weights/weights_;
        m = 1;
        for j = 1:rest+1
            if index_ref(j,i) ~= 1
                index_ref(j,i) = weights(m);
                m = m + 1;
            else
                index_ref(j,i) = 0;
            end
        end
    end
end
index = index_ref(1:end-1,:);
demands = Dtrain;
% Except for demands information, we need index to indicate which data
% belong to which demands
% The data structure I used in nnnewsvendorloss_MTL.m is 
%  demands:   1 x N vector, such that demands(i) is the demand quantity for
%             the i-th case (data x(i))
%  index:     d x N matrix of 0 and 1.   Its i-th column is a 0-1 vector
%             with only one 1. The location of this 1 means which product
%             You must construct this index matrix according to your data
%             
% More comments: 
% I check you data above, according to IDs, it seems there are 400 different products.
% I would like to suggest you meage 400 products into, e.g., two categories
% and then construct index accordingly. However I dont have good suggestion
% on merging.  Do the original data come with dates?  If so, you can add
% all the products of each of your chosen categories together as one datum for the
% day.
%
% I still find it seems to me you need sharpen your programming skills too.
%% Define cp and ch
ch = rand(1, O) + 1;
cp = 3 * ch;
%% Building the network   Junbin Gao
net.layers={} ;
f=1/10 ; 
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
                       
% lambda for weights regularization
lambda = 0.001;

% Beta is not used
beta = 0;

% delta for weights on products
delta = rand(O);
delta_ = sum(delta);
delta = delta / delta_;
delta = delta';

theta = [W1(:); B1(:); W2(:); B2(:); Wo(:); Bo(:)];

%% Junbin's comments   15 August 2017
% Now in the last layer (i.e., newsvendorloss_MTL) we all need store the index
% information.  You may add 'index', index  after demands in the following
% statement
% Thus we need to change the calculate in newsvendorCost.m function for
% this new MTL loss layer
net.layers{end+1} = struct('type', 'newsvendorloss_MTL_new', ...
                            'ch', ch, 'cp', cp, 'demands', demands , 'index', index, 'label_tag', Ltrain) ;

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
options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.MaxFunEvals = 1000;	

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
%% optimal training error
[optTrainErr,~] = newsvendorCost_new(opttheta, net, XtrainB, lambda, beta, delta);
%% TestError
% [~,test_]=size(Dtest);
% demands=Dtest;
% index_ = zeros(O, test_);
% count_start = 1;
% for i = 1:O
%     if i ~= O
%         count_end = fix(count_start + test_/O);
%         index_(i, count_start:count_end) = ones(1, count_end - count_start+1);
%     else
%         index_(i, count_end:test_) = ones(1, test_ - count_end+1);
%     end
% end
% net.layers{end} = struct('type', 'newsvendorloss_MTL', ...
%                             'ch', ch, 'cp', cp, 'demands', demands , 'index', index_) ;
% [TestErr,~] = newsvendorCost_new(opttheta, net, XtestB, lambda, beta, delta);

%% Prediction
[~,test_]=size(Dtest);
sead = randi([1,test_], 100, 1);
Dtest(1, sead) = -1;
index_modified = [];
for i = 1:test_
    if i ~= sead
        index_modified(1,end+1) = 1;
    else
        index_modified(1,end+1) = 0;
    end
end

res = SimpleNN(net, XtestB, [], [], 'prediction', 1, 'beta', beta);
predicted_demand = res(L).x;
predicted_demand_modified = zeros(1, test_);
count_start = 1;
for i = 1:O
    if i ~= O
        label = length(find(Ltest == i));
        count_end = count_start + label;
        predicted_demand_modified(count_start:count_end) = predicted_demand(i, count_start:count_end);
        count_start = count_end;
    else
        label = length(find(Ltest == i));
        count_end = count_start + label - 1;
        predicted_demand_modified(count_start:count_end) = predicted_demand(i, count_start:count_end);
    end
end
TestErr = index_modified .* (Dtest - predicted_demand_modified).^2;
TestErr = mean(TestErr);