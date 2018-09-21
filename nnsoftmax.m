function Y = nnsoftmax(X,dzdY)
%NNSOFTMAX softmax.
%   Y = NNSOFTMAX(X) applies the softmax operator the data X. X
%   has dimension D x N, packing N  
%   D-dimensional vectors.
%
%   D can be thought of as the number of possible classes and the
%   function computes the softmax along the D dimension.  
%
%   DZDX = NNSOFTMAX(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.

% This code is adopted from MatConvNet by Professor Junbin Gao
%
% The original code is part of the VLFeat library  Copyright (C) 2014 Andrea Vedaldi
% under the terms of the BSD license. All rights reserved.

E = exp(bsxfun(@minus, X, max(X,[],1))) ;
L = sum(E,3) ;
Y = bsxfun(@rdivide, E, L) ;

if nargin <= 1, return ; end

% backward
Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y, 1)) ;
