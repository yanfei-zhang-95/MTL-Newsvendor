function y = nnnewsvendorloss_MTL_new(x, demand, cp, ch, index, label_tag, dzdy, varargin)
%   Professor Junbin Gao revise from the new version of nnnewsvendorloss.m to this
%   file.  15 August 2017 
%
%   Y = NNNEWSVENDORLOSS_MTL(x, demand, cp, ch) calculates the final cost with two
%   the rectified linear units for the data X. X can have arbitrary size 
%   where the last dimension is the number of data, most 
%   likely both X and TARGET are d x N (d products d demands) and demand is a 1 x N vector
%   where the i-th component of demand is the total number for the j-th
%   product, which is determined by Hot-One code index, for example (1, 0,
%   0, ...,0)' means that this data belongs to the first product etc.
%
%   We assume both cp and ch are scalars or vectors of length d.  If they
%   are scalars, these will be applied to all the demands for products.
%
%   DZDX = NNNEWSVENDORLOSS(x, demand, b, h, DZDY) computes the derivative 
%   of the block projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and TARGET respectively.
%
%   NNRELU(...,'OPT',VALUE,...) takes the following options:
%
%   `Leak`:: 0
%      Set the leak factor, a non-negative number. Y is equal to X if
%      X is not smaller than zero; otherwise, Y is equal to X
%      multipied by the leak factor. By default, the leak factor is
%      zero; for values greater than that one obtains the leaky ReLU
%      unit.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = NNRELU(X), then
%   NNRELU(X,DZDY) gives the same result as NNRELU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.

% This code is adopted from MatConvNet by Professor Junbin Gao
%
% The original code is part of the VLFeat library  Copyright (C) 2014 Andrea Vedaldi
% under the terms of the BSD license. All rights reserved.

opts.leak = 0 ;
opts = argparse(opts, varargin) ;

d = size(x, 1);
N = size(x, 2);

if numel(cp) == 1
    cp = cp * ones(d, 1);
end
if numel(ch) == 1
    ch = ch * ones(d, 1);
end

assert(numel(cp)==d, 'cp dimension should be equal to number of products')
assert(numel(ch)==d, 'ch dimension should be equal to number of products')
cp = cp(:);   % making sure it is in column
ch = ch(:); 
[~ , n] = size(demand);

D = cell(1,d);
for i = 1:d
   D{i} = demand(find(label_tag == i));
end

mtx = zeros(d, n);
for i = 1:d
    mtx(i, :) = repmat(D{i},1,d);
end

demand = mtx;

%% Failed Tries on demand indexing
% mtx = zeros(d, n);
% for i = 1:d
%     count_start = 1;
%     label = length(D{i});
%     count_end = count_start + label;
%     mtx(i, count_start:count_end-1) = D{i};
%     count_start = count_end;
%     for j = 1:d
%         if j ~= i
%             label = length(D{j});
%             count_end = count_start +label;
%             mtx(i, count_start:count_end-1) = D{j};
%             count_start = count_end;
%         end
%     end
% end
% 
% demand = mtx;
%%

if opts.leak == 0
  if nargin <= 6 || isempty(dzdy)
    % Here we are using the trick for convenient calculation
    temp = index .* (repmat(cp, 1, N) .* max(demand - x, 0.0) + repmat(ch, 1, N) .* max ( x - demand, 0.0));  % calculate on each component  
    y = sum(temp(:));
  else
    y = dzdy .* ((- repmat(cp, 1, N) .* ((demand - x) > 0.0) + repmat(ch, 1, N).*((demand - x) < 0.0)) .* index) ;
  end
else   % We don't touch this
  if nargin <= 6 || isempty(dzdy)
    %y = x .* (opts.leak + (1 - opts.leak) * single(x > 0)) ;
    
    y = sum(cp * ((demand - x) .*  (opts.leak + (1 - opts.leak) * ( demand - x > 0.0 ))) + ...
        ch * ((x - demand) .*  (opts.leak + (1 - opts.leak) * ( x - demand > 0.0 ))));
  else
    y = dzdy .* ( -cp * (opts.leak + (1 - opts.leak) * (demand - x > 0.0 ))  + ...
                   ch * (opts.leak + (1 - opts.leak) * (x - demand > 0.0 ))) ; 
  end
end
