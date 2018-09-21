function [cost,grad] = newsvendorCost_trace(theta, net, data, lambda, gamma, beta)

cost = 0;
grad = zeros(size(theta));
L = numel(net.layers);
[~, N] = size(data); 

count_end = 0;
for i = 1:L
    if i<L-1
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
    else
        if isfield(net.layers{i}, 'weights')
        m = length(net.layers{i}.weights);
            for j=1:m
                count_start = count_end + 1;
                w_size = size(net.layers{i}.weights{j});
                count_end = count_start + prod(w_size) - 1;
                net.layers{i}.weights{j} = reshape(theta(count_start:count_end), w_size);
                if j < m 
                    [U, S, V] = svd(net.layers{i}.weights{j});
                    sing = diag(S);
                    cost = cost + gamma * sum(sing);
                end
            end
        end 
    end
end

res = SimpleNN(net, data, 1, [], 'beta', beta);

cost = cost+res(end).x/N;
count_end = 0;
for i = 1:L
    if i < L-1
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
    else
        if isfield(net.layers{i}, 'weights')
            m = length(net.layers{i}.weights);
                for j=1:m
                    count_start = count_end + 1;
                    w_size = size(net.layers{i}.weights{j});
                    count_end = count_start + prod(w_size) - 1;
                    dzdw = res(i).dzdw{j};
                    grad(count_start:count_end) = dzdw(:)/N;
                    grad_=reshape(grad(count_start:count_end), w_size(1), w_size(2));
                    if j < m
                        [U, ~, V] = svds(net.layers{i}.weights{j}, w_size(1));
                        grad_ = grad_ + gamma*(U*V');
                    end
                    grad(count_start:count_end)=grad_(:);
                end
        end
    end 
end

end

