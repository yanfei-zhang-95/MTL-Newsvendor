function [cost,grad] = newsvendorCost(theta, net, data, lambda, beta)
 
% Copyright (C) 2017 Professor Junbin Gao.
% All rights reserved.
%
% This file is part of the NN library and is made available under
% the terms of the BSD license (see the COPYING file).

cost = 0;
grad = zeros(size(theta));
L = numel(net.layers);
[~, N] = size(data); 

count_end = 0;
for i = 1:L
    if isfield(net.layers{i}, 'weights')
        m = length(net.layers{i}.weights);
        for j=1:m
            count_start = count_end + 1;
            w_size = size(net.layers{i}.weights{j});
            count_end = count_start + prod(w_size) - 1;
            net.layers{i}.weights{j} = reshape(theta(count_start:count_end), w_size);
            if j < m
                cost = cost + 0.5*lambda*sum(reshape(net.layers{i}.weights{j},[],1).^2);
            end
        end
    end    
end

res = SimpleNN(net, data, 1, [], 'beta', beta);

% for i = 1:L
%     if strcmp(net.layers{i}.type, 'sparse')  
%        cost = cost + res(i).aux; 
%     end    
% end;

cost = cost+res(end).x/N;
count_end = 0;
for i=1:L    
    if isfield(net.layers{i}, 'weights')
        m = length(net.layers{i}.weights);
        for j=1:m
            count_start = count_end + 1;
            w_size = size(net.layers{i}.weights{j});
            count_end = count_start + prod(w_size) - 1;
            dzdw = res(i).dzdw{j};
            grad(count_start:count_end) = dzdw(:)/N;
            if j < m
                grad(count_start:count_end) = grad(count_start:count_end) + lambda*reshape(net.layers{i}.weights{j},[],1);
            end
        end
    end
end

end

