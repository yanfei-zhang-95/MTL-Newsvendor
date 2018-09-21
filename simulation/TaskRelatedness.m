clc;
clear;
close;
%%
load NewsvendorData
%%
mu = ones(1, 5);
sigma = 0.5:0.1:0.9;
%%
sigma = sigma'*sigma;
rng default  % For reproducibility
w1 = mvnrnd(mu,sigma,43)';
%%
Dtrain = w1 * XtrainB;
Dtrain = [Dtrain; rand(20,9877)];