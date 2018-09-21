clear all
rng('default')
rng(0)

N = 100;
D = 4;
d = 3;

x  = randn(D,N);
W = randn(d, D);
B = rand(d,1);

y = nnlinear(x, W, B, []);
px = ones(size(y));
px = rand(size(y));
[dx, dw, db] = nnlinear(x, W, B, px);

funcx = @(xx) tnnproj(px, nnlinear(xx, W, B, [])) ; 
err = tnncheckDerivativeNumerically(funcx, x, dx) ;
disp(err)

funcW = @(ww) tnnproj(px, nnlinear(x, ww, B, [])) ;
errW = tnncheckDerivativeNumerically(funcW, W, dw) ;
disp(errW)

funcB = @(bb) tnnproj(px, nnlinear(x, W, bb, [])) ;
errB = tnncheckDerivativeNumerically(funcB, B, db) ;
disp(errB)


%%% I have revised nnnewsvendorloss.m to work for multiple product demands
% So I test it still works for the old problem (single product)
cp = 2;
ch = 3;
dzdy = 1;
x  = randn(1,N);
demand = randn(1, N);
dx = nnnewsvendorloss(x, demand, cp, ch, dzdy) ;

funcx = @(xx) nnnewsvendorloss(xx, demand, cp, ch, []) ; 
err = tnncheckDerivativeNumerically(funcx, x, dx) ;
disp(err)


% This is to test newly revised function works well for derivative
%  15 August 2017   Professor Junbin Gao
cp = 2;
ch = 3;
dzdy = 1;
x  = randn(2,N);
demand = randn(2, N);
dx = nnnewsvendorloss(x, demand, cp, ch, dzdy) ;

funcx = @(xx) nnnewsvendorloss(xx, demand, cp, ch, []) ; 
err = tnncheckDerivativeNumerically(funcx, x, dx) ;
disp(err)

% Multiple cp and ch
cp = [2, 2.1];
ch = [3, 3.1];
dzdy = 1;
x  = randn(2,N);
demand = randn(2, N);
dx = nnnewsvendorloss(x, demand, cp, ch, dzdy) ;

funcx = @(xx) nnnewsvendorloss(xx, demand, cp, ch, []) ; 
err = tnncheckDerivativeNumerically(funcx, x, dx) ;
disp(err)
%%
% Testing Multiple Task Loss Junbin Gao
cp = [2, 2.1];
ch = [3, 3.1];
dzdy = 1;
x  = randn(2,N);
demand = randn(1, N);
% Here we produce N indices to indicate which data belong to which product.
% (1, 0)^T means the first product and (0, 1)^T means the second product.
% The current programs work for any number of d products.
index = randi([0,1], 1, 2*N);
index = reshape(index, 2, N);
for i = 1:N
    if (index(1, i) == 1) && (index(2,i)==1)
        index(1,i) = 0;
    end
end
for i = 1:N
    if (index(1, i) == 0) && (index(2,i)==0)
        index(1,i) = 1;
    end
end
dx = nnnewsvendorloss_MTL(x, demand, cp, ch,index, dzdy) ;

funcx = @(xx) nnnewsvendorloss_MTL(xx, demand, cp, ch, index, []) ; 
err = tnncheckDerivativeNumerically(funcx, x, dx) ;
disp(err)