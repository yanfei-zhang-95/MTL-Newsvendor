function y = nnrelu(x,dzdy,varargin)
%NNRELU rectified linear unit.
%   Y = NNRELU(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = NNRELU(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%   VL_NNRELU(...,'OPT',VALUE,...) takes the following options:
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

if opts.leak == 0
  if nargin <= 1 || isempty(dzdy)
    y = max(x, single(0)) ;
  else
    y = dzdy .* (x > single(0)) ;
  end
else
  if nargin <= 1 || isempty(dzdy)
    y = x .* (opts.leak + (1 - opts.leak) * single(x > 0)) ;
  else
    y = dzdy .* (opts.leak + (1 - opts.leak) * single(x > 0)) ; 
  end
end
