function y = nnnewsvendorloss01(x, demand, cp, ch, index, dzdy, varargin)
%NNRELU rectified linear unit.
%   Y = NNNEWSVENDORLOSS(x, demand, b, h) calculates the final cost with two
%   the rectified linear units for the data X. X can have arbitrary size, most 
%   likely both X and TARGET are in in 1 x N 
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
cp = cp(:);
ch = ch(:);

if opts.leak == 0
  if nargin <= 5 || isempty(dzdy)
    temp = sum(index .* (repmat(cp,1,N) .* max(demand - x, 0.0) + repmat(ch,1,N) .* max(x - demand, 0.0)));
    y = sum(temp(:));
  else
    y = dzdy .* (index .* ( - repmat(cp,1,N) .* ((demand - x) > 0.0) + repmat(ch,1,N) .* ( (demand - x) < 0.0))) ;
  end
else
  if nargin <= 4 || isempty(dzdy)
    %y = x .* (opts.leak + (1 - opts.leak) * single(x > 0)) ;
    
    y = sum(cp * ((demand - x) .*  (opts.leak + (1 - opts.leak) * ( demand - x > 0.0 ))) + ...
        ch * ((x - demand) .*  (opts.leak + (1 - opts.leak) * ( x - demand > 0.0 ))));
  else
    y = dzdy .* ( -cp * (opts.leak + (1 - opts.leak) * (demand - x > 0.0 ))  + ...
                   ch * (opts.leak + (1 - opts.leak) * (x - demand > 0.0 ))) ; 
  end
end
