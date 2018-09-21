function [y, dzdw, dzdb] = nnlinear(x, W, B, dzdy, varargin)
%NNLINEAR linear mapping unit.
%   Y = NNLINEAR(x, W, B) calculates the linear mapping between two layers.
%   X is assume in dimension D x N;  Y in d x N;  W is the coefficient in d x D
%   and B is the bias vector of dimension d   
%
%   DZDX = NNNEWSVENDORLOSS(x, W, B, DZDY) computes the derivative 
%   of the block projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%   ADVANCED USAGE
%

% This code is adopted from MatConvNet by Professor Junbin Gao
%
% The original code is part of the VLFeat library  Copyright (C) 2014 Andrea Vedaldi
% under the terms of the BSD license. All rights reserved.

%opts.leak = 0 ;
%opts = argparse(opts, varargin) ;

 
if nargin <= 3 || isempty(dzdy)
   y = bsxfun(@plus, W * x, B);
else
   y = W' * dzdy;
   dzdw = dzdy * x';
   dzdb = sum(dzdy,2);
end
 