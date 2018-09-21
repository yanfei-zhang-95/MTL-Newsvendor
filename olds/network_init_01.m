function net = network_init_01(varargin)
% Networks Initialize a newvendor problem

opts = argparse(opts, varargin) ;

rng('default');
rng(0) ;

 

f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{f*randn(10, 20), zeros(10,1)}}) ;   % First hidden layer

net.layers{end+1} = struct('type','sigmoid');
 
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{f*randn(15,10), zeros(15,1)}}) ;
net.layers{end+1} = struct('type','sigmoid');

% Output to a single estimate for demand
net.layers{end+1} = struct('type', 'linear', ...
                           'weights', {{10*f*randn(1,15), zeros(1,1)}}) ;
 
 
net.layers{end+1} = struct('type', 'newsvendorloss', ...
                            'ch', 2, 'cp', 1) ;
 
 