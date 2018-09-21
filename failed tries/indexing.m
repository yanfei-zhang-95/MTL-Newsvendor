clc;
clear all;
close all;
%%
load NewsvendorData_Label_;
%%
[~, n] = size(XtrainB);
%%
D = size(XtrainB, 1);%Number of input features
[~, O] = size(unique(Ltrain));
demands = Dtrain;
[~,train_] = size(Dtrain);
%%
index = zeros(O, train_);
count_start = 1;
for i = 1:O
    label = length(find(Ltrain == i));
    count_end = count_start + label;
    index(i, count_start:count_end) = ones(1, count_end - count_start + 1);
    count_start = count_end;
end
index = index(:, 1:train_);
%%
seed = randi([1,n], 1000,1);
Dtrain(1, seed) = -1;
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