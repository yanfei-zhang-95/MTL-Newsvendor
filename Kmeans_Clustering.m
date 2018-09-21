clc;
clear all;
close all;
%% Kmeans Clustering
load NewsvendorData;
D_ = [Dtrain Dtest; XtrainB XtestB]';
[~,N] = size(D_);
k = 10;
cls = kmeans(D_(2:end,:), k);
% cls = kmeans(D_, k);
%% Label
D = cell(1, k);
Dtrain_ = [];
Dtest_ = [];

for i = 1:k
    D{i} = D_(find(cls == i),:)';
    [~,n] = size(D{i});
    D{i} = [i * ones(1, n); D{i}];
    train = D{i}(:, 1:fix(0.71 * n));
    Dtrain_ = [Dtrain_ train];
    test = D{i}(:, fix(0.71 * n)+1: end);
    Dtest_ =  [Dtest_ test];
end

%%
Ltrain = Dtrain_(1,:);
Ltest = Dtest_(1,:);
for i = unique(Ltrain)
    [~, q] = size(find(Ltrain == i));
    Ltrain(:, find(Ltrain == i)) = i * ones(1, q);
end
for i = unique(Ltest)
    [~, q] = size(find(Ltest == i));
    Ltest(:, find(Ltest == i)) = i * ones(1, q);
end
%%
Dtrain = Dtrain_(2,:);
Dtest = Dtest_(2,:);
XtrainB = Dtrain_(3:end,:);
XtestB = Dtest_(3:end,:);
save('NewsvendorData_Label','Ltrain','Ltest','Dtrain','Dtest','XtrainB','XtestB');