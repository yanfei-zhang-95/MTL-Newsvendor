function res = SimpleNN(net, x, dzdy, res, varargin)
% SIMPLENN  Evaluates a simple NN
%   RES = SIMPLENN(NET, X) evaluates the NET on data X.
%   RES = SIMPLENN(NET, X, DZDY) evaluates the NET and its
%   derivative on data X and output derivative DZDY.
%
%   The network has a simple (linear) topology, i.e. the computational
%   blocks are arranged in a sequence of layers. Please note that
%   there is no need to use this wrapper, which is provided for
%   convenience. Instead, the individual CNN computational blocks can
%   be evaluated directly, making it possible to create significantly
%   more complex topologies, and in general allowing greater
%   flexibility.
%
%   The NET structure contains two fields:
%
%   - net.layers: the NN layers.
%   - net.normalization: information on how to normalize input data.
%
%   The network expects the data X to be already normalized. This
%   usually involves rescaling the input image(s) and subtracting a
%   mean.
%
%   RES is a structure array with one element per network layer plus
%   one representing the input. So RES(1) refers to the zeroth-layer
%   (input), RES(2) refers to the first layer, etc. Each entry has
%   fields:
%
%   - res(i+1).x: the output of layer i. Hence res(1).x is the network
%     input.
%
%   - res(i+1).aux: auxiliary output data of layer i. For example,
%     dropout uses this field to store the dropout mask.
%
%   - res(i+1).dzdx: the derivative of the network output relative to
%     variable res(i+1).x, i.e. the output of layer i. In particular
%     res(1).dzdx is the derivative of the network output with respect
%     to the network input.
%
%   - res(i+1).dzdw: the derivative of the network output relative to
%     the parameters of layer i. It can be a cell array for multiple
%     parameters.
%
%   net.layers is a cell array of network layers. The following
%   layers, encapsulating corresponding functions in the toolbox, are
%   supported:
%
 
%   Normalization layer::
%     The normalization layer wraps NNNORMALIZE(). It has fields
%
%     - layer.type = 'normalize'
%     - layer.param: the normalization parameters.
%
 
%   ReLU and Sigmoid layers::
%     The ReLU layer wraps NNRELU(). It has fields:
%
%     - layer.type = 'relu'
%
%     The sigmoid layer is the same, but for the sigmoid function, with
%     `relu` replaced by `sigmoid`.
%
%   Dropout layer::
%     The dropout layer wraps NNDROPOUT(). It has fields:
%
%     - layer.type = 'dropout'
%     - layer.rate: the dropout rate.
%
%   Softmax layer::
%     The softmax layer wraps NNSOFTMAX(). It has fields
%
%     - layer.type = 'softmax'
%
%   Log-loss layer::
%     The log-loss layer wraps NNLOSS(). It has fields:
%
%     - layer.type = 'loss'
%     - layer.class: the ground-truth class.
%
%   Softmax-log-loss layer::
%     The softmax-log-loss layer wraps NNSOFTMAXLOSS(). It has
%     fields:
%
%     - layer.type = 'softmaxloss'
%     - layer.class: the ground-truth class.
%
%   P-dist layer::
%     The pdist layer wraps NNPDIST(). It has fields:
%
%     - layer.type = 'pdist'
%     - layer.p = P parameter of the P-distance
%     - layer.noRoot = whether to raise the distance to the P-th power
%     - layer.epsilon = regularization parameter for the derivatives
%
%   Custom layer::
%     This can be used to specify custom layers.
%
%     - layer.type = 'custom'
%     - layer.forward: a function handle computing the block.
%     - layer.backward: a function handle computing the block derivative.
%
%     The first function is called as res(i+1) = forward(layer, res(i), res(i+1))
%     where res() is the struct array specified before. The second function is
%     called as res(i) = backward(layer, res(i), res(i+1)). Note that the
%     `layer` structure can contain additional fields if needed.

% This code is adopted from MatConvNet by Professor Junbin Gao
%
% The original code is part of the VLFeat library  Copyright (C) 2014 Andrea Vedaldi
% under the terms of the BSD license. All rights reserved.


opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.beta = 0.0;
opts.prediction = 0;

opts = argparse(opts, varargin);


n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;
L = numel(net.layers);

for i=1:(n - opts.prediction)
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type 
    case 'linear'
      res(i+1).x = nnlinear(res(i).x,l.weights{1}, l.weights{2});
      if i == L-1
          weight_ = l.weights{1};
      end
    case 'newsvendorloss'
      res(i+1).x = nnnewsvendorloss(res(i).x, l.demands, l.cp, l.ch);
    case 'newsvendorloss_MTL'
      res(i+1).x = nnnewsvendorloss_MTL(res(i).x, l.demands, l.cp, l.ch, l.index);
    case 'newsvendorloss_MTL_new'
      res(i+1).x = nnnewsvendorloss_MTL_new(res(i).x, l.demands, l.cp, l.ch, l.index, weight_);
    case 'newsvendorloss_l2'
      res(i+1).x = nnnewsvendorloss_l2(res(i).x, l.demands, l.cp, l.ch);    
    case 'softmax'
      res(i+1).x = nnsoftmax(res(i).x) ;  
    case 'relu'
      if isfield(l, 'leak'), leak = {'leak', l.leak} ; else leak = {} ; end
      res(i+1).x = nnrelu(res(i).x,[],leak{:}) ;
    case 'sigmoid'
      res(i+1).x = nnsigmoid(res(i).x) ;
    case 'dropout'
      if opts.disableDropout
        res(i+1).x = res(i).x ;
      elseif opts.freezeDropout
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end
    case 'pdist'
      res(i+1) = vl_nnpdist(res(i).x, l.p, 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  if forget
    res(i).x = [] ;
  end
%   if gpuMode & opts.sync
%     % This should make things slower, but on MATLAB 2014a it is necessary
%     % for any decent performance.
%     wait(gpuDevice) ;
%   end
  res(i).time = toc(res(i).time) ;
end

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'linear' 
        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = nnlinear(res(i).x,l.weights{1}, l.weights{2}, res(i+1).dzdx); 
      case 'newsvendorloss'
        res(i).dzdx = nnnewsvendorloss(res(i).x, l.demands, l.cp, l.ch, res(i+1).dzdx);     
      case 'newsvendorloss_MTL'
        res(i).dzdx = nnnewsvendorloss_MTL(res(i).x, l.demands, l.cp, l.ch, l.index, res(i+1).dzdx);
      case 'newsvendorloss_MTL_new'
        res(i).dzdx = nnnewsvendorloss_MTL_new(res(i).x, l.demands, l.cp, l.ch, l.index, l.label_tag, res(i+1).dzdx);
      case 'newsvendorloss_l2'
        res(i).dzdx = nnnewsvendorloss_l2(res(i).x, l.demands, l.cp, l.ch, res(i+1).dzdx);     
      case 'softmax'
        res(i).dzdx = nnsoftmax(res(i).x, res(i+1).dzdx) ; 
      case 'relu'
        if isfield(l, 'leak'), leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          res(i).dzdx = nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          res(i).dzdx = nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end
      case 'sigmoid'
        res(i).dzdx = nnsigmoid(res(i).x, res(i+1).dzdx) ;
      case 'dropout'
        if opts.disableDropout
          res(i).dzdx = res(i+1).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                                     'mask', res(i+1).aux) ;
        end
      case 'pdist'
        res(i).dzdx = vl_nnpdist(res(i).x, l.p, res(i+1).dzdx, ...
                                 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
%     if gpuMode & opts.sync
%       wait(gpuDevice) ;
%     end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end

