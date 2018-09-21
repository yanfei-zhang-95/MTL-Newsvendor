function err = tnncheckDerivativeNumerically(f, x, dx)
%TNNCHECKDERIVATIVENUMERICALLY  Check a tensorial layer's deriviative numerically
%   ERR = TNNCHECKDERIVATIVENUMERICALLY(F, X, DX) takes the scalar function F,
%   its tensor input X and its derivative DX at X and compares DX to
%   a numerical approximation of the derivative returing their difference
%   ERR.

% Copyright (C) 2016 Professor Junbin Gao.
% All rights reserved.
%
% This file is part of the tNN library and is made available under
% the terms of the BSD license (see the COPYING file).


%y = f(x) ;

delta = 1e-5 ;

dimsX = size(x); 
%D = length(dimsX);   % Data dimension;
N = prod(dimsX);   % Total number of variables
dx_numerical = zeros(N, 1) ;
x = x(:);
dx = dx(:);

for i = 1:N
    xp = x;
    xp(i) = xp(i) + delta;
    xp = reshape(xp, dimsX);
    xn = x;
    xn(i) = xn(i) - delta;
    xn = reshape(xn, dimsX);
    yn = f(xn);
    yp = f(xp);
    dx_numerical(i) = (yp - yn) / (2*delta) ;
end

dx_numerical_non_zeros=[0 0];
for i=1:N
    if dx_numerical(i)~=0
        dx_numerical_non_zeros(end+1,:)=[i dx_numerical(i)];
    end
end

% save("dx_numerical_non_zeros","dx_numerical_non_zeros");

err = norm(dx_numerical - dx)/norm(dx_numerical + dx);

% ind_str = 'indx1';
% for i = 2:D
%     ind_str = [ind_str, ', indx', num2str(i)];
% end
% ind_str = ['[', ind_str, ']'];
% 
% N = prod(dimsX);   % Total number of variables
% for i = 1:N
%     eval([ind_str, ' = ind2sub(dimsX,', num2str(i), ');']);
%     xp = x;
%     eval(['xp(', ind_str, ') = xp(', ind_str, ') + delta;']);
%     yp = f(xp);
%     xn = x;
%     eval(['xn(', ind_str, ') = xn(', ind_str, ') - delta;']);
%     yn = f(xn);
%     eval(['dx_numerical(', ind_str, ') = (yp - yn) / (2*delta) ;']);
% end

% n = length(dimsX);   % dimension of x
% for n = 1:size(x,4)
%   for k = 1:size(x,3)
%     for j = 1:size(x,2)
%       for i = 1:size(x,1)
%         xp = x ;
%         xp(i,j,k,n) = xp(i,j,k,n) + delta ;
%         yp = f(xp) ;
%         dx_numerical(i,j,k,n) =  (yp - y) / delta ;
%       end
%     end
%   end
% end
%err = dx_numerical - dx ;
% err = norm(dx_numerical(:) - dx(:))/norm(dx_numerical(:) + dx(:));
%disp(diff)


% range = max(abs(dx(:))) * [-1 1] ;
% T = min(3, N);  
% indx = randi(T, 4,1);
% for t = 1:3
%   subplot(3,3,1+(t-1)*3) ; bar3(dx(:,:,1,t)) ; zlim(range) ;
%   title(sprintf('dx(:,:,1,%d) (given)',t)) ;
%   subplot(T,3,2+(t-1)*3) ; bar3(dx_numerical(:,:,1,t)) ; zlim(range) ;
%   title(sprintf('dx(:,:,1,%d) (numerical)',t)) ;
%   subplot(T,3,3+(t-1)*3) ; bar3(abs(err(:,:,1,t))) ; zlim(range) ;
%   title('absolute difference') ;
% end
