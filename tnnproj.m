function z = tnnproj(x,p)
%TNNPROJ  Project a tensor onto another
%   Z = TNNPROJ(X,P) computes the projection Z of tensor X onto P.
%
%   Remark: if X and P contain multiple tensor instances
%   (concatenated along the foruth dimension), then the
%   result Z contains a scalar projection for each.


% Copyright (C) 2016 Professor Junbin Gao.
% All rights reserved.
%
% This file is part of the tNN library and is made available under
% the terms of the BSD license (see the COPYING file).


prods = x .* p ;
z = sum(prods(:)) ;
