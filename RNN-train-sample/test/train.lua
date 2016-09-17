-- Eugenio Culurciello
-- September 2016
-- RNN training test: ABBA sequence detector
-- based on: https://raw.githubusercontent.com/karpathy/char-rnn/master/model/RNN.lua

require 'nn'
require 'nngraph'
require 'optim'
require 'oneHot'
dofile('RNN.lua')
function pr(...) print(tostring(...)) end
-- local c = require 'trepl.colorize'
local model_utils = require 'model_utils'

torch.setdefaulttensortype('torch.FloatTensor')
-- nngraph.setDebug(true)

local opt = {}
rho = 4 
nIndex = 2
trainSize = 1000
nClass = 2
batchSize = 1
opt.dictionary_size = nIndex -- sequence of 2 symbols
opt.train_size = trainSize -- train data size
opt.seq_length = rho -- RNN time steps

print('Creating Input...')
-- build input
-- create a sequence of 2 numbers: {2, 1, 2, 2, 1, 1, 2, 2, 1 ..}
ds = {}
ds.size = trainSize 
local target_seq = torch.LongTensor({1,2,2,1})
ds.input = torch.LongTensor(ds.size,rho):random(nClass)
--ds.input = torch.expand(target_seq:resize(1,4), trainSize, rho)
ds.target = torch.LongTensor(ds.size):fill(1)
-- initialize targets:
local indices = torch.LongTensor(rho)
for i=1, ds.size do
  if torch.sum(torch.abs(torch.add(ds.input[i], -target_seq))) == 0 then ds.target[i] = 2 end
end

indices:resize(batchSize)
pr('indices')
pr(indices)
pr('target_seq')
pr('input: ')
pr(ds.input)
pr('output')
pr(ds.target)
-- model:
print('Creating Model...')
opt.rnn_size = 10 
opt.rnn_layers = 1
opt.batch_size = 1

local protos = {} 
protos.rnn = RNN(opt.dictionary_size, opt.rnn_size, opt.rnn_layers, 0) -- input = 2 (classes), 1 layer, rnn_size=1, no dropout 
protos.criterion = nn.ClassNLLCriterion()
protos.oneHot = OneHot(nClass)
-- print('Test of RNN output:', RNNmodel:forward{ torch.Tensor(2), torch.Tensor(1) })

-- the initial state of the cell/hidden states
local init_state = {}
for L = 1, opt.rnn_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
  table.insert(init_state, h_init:clone())
end

local params, grad_params 
-- get flattened parameters tensor
-- params, grad_params = model_utils.combine_all_parameters(protos.rnn)
params, grad_params = protos.rnn:getParameters()
print('Number of parameters in the model: ' .. params:nElement())
-- print(params, grad_params)

-- create clones of model to unroll in time:
print('Cloning RNN model:')
local clones = {}
clones.rnn = {}
clones.criterion = {}
for i = 1,opt.seq_length do
   clones.rnn[i] = protos.rnn:clone('weight', 'gradWeights', 'gradBias', 'bias')
   clones.criterion[i] = protos.criterion:clone()
end
-- for name, proto in pairs(protos) do
--   print('cloning ' .. name)
--   clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
-- end


function clone_list(tensor_list, zero_too)
  -- takes a list of tensors and returns a list of cloned tensors
  local out = {}
  for k,v in pairs(tensor_list) do
      out[k] = v:clone()
      if zero_too then out[k]:zero() end
  end
  return out
end

-- training function:
local init_state_global = clone_list(init_state)
local bo = 0 -- batch offset / counter
opt.grad_clip = 0.08 
-- training:
opt.learning_rate = 0.001
opt.decay_rate = 0.95
print('Training...')
local losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = ds.size
for i = 1, iterations do
   local index  = math.fmod(i-1,ds.size)+1
   function feval(p)
     if p ~= params then
       params:copy(p)
     end
     grad_params:zero()

     -- bo variable creates batches on the fly
     
     -- forward pass ---------------------------------------------------------------
     local rnn_state = {[0]=init_state_global} -- initial state
     local predictions = {}
     local loss = 0
     local inPut, tarGet
     for t = 1, opt.seq_length do
       inPut = protos.oneHot:forward(ds.input[index][t])
       tarGet = protos.oneHot:forward(ds.target[index])
       clones.rnn[t]:training() -- make sure we are in correct training mode
       lst = clones.rnn[t]:forward({inPut, unpack(rnn_state[t-1])})
       rnn_state[t] = {}
       for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
       predictions[t] = lst[#lst]
       local lst
       if t == opt.seq_length then
          loss = loss + clones.criterion[t]:forward(predictions[t], ds.target[index])
       end
     end
     loss = loss / opt.seq_length
     -- backward pass --------------------------------------------------------------
     -- initialize gradient at time t to be zeros (there's no influence from future)
     local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
     local dlst
     for t = opt.seq_length, 1, -1 do
       -- print(drnn_state)
       -- backprop through loss, and softmax/linear
       -- print(predictions[t], y[{1,{t+bo}}])
       local doutput_t 
       if t == opt.seq_length then
          doutput_t = clones.criterion[t]:backward(predictions[t], ds.target[i])
          table.insert(drnn_state[t], doutput_t)
          dlst = clones.rnn[t]:backward({inPut, unpack(rnn_state[t-1])}, drnn_state[t])
       else
          dlst = clones.rnn[t]:backward({inPut, unpack(rnn_state[t-1])}, drnn_state[t])
       end
       -- print('douptut', doutput_t)
       drnn_state[t-1] = {}
       for i = 1, #dlst do
          local j = #dlst+1 - i
          drnn_state[t-1][j] = dlst[i]
       end
     end
     -- transfer final state to initial state (BPTT)
     --init_state_global = rnn_state[#rnn_state]
     -- grad_params:div(opt.seq_length)
     -- clip gradient element-wise
     grad_params:clamp(-opt.grad_clip, opt.grad_clip)
     -- print(params,grad_params)
     -- point to next batch:
     return loss, grad_params
   end


  local _, loss = optim.rmsprop(feval, params, optim_state)
  losses[#losses + 1] = loss[1]

  if i % (iterations/10) == 0 then
    print(string.format("Iteration %8d, loss = %4.4f, loss/seq_len = %4.4f, gradnorm = %4.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
  end
end


-- testing function:
local perm = torch.randperm(ds.size)
function test(inDex)
  -- bo variable creates batches on the fly
  
  -- forward pass ---------------------------------------------------------------
  local rnn_state = {[0]=init_state_global} -- initial state
  local predictions
  local loss = 0
  local inPut
  for t = 1, opt.seq_length do
    clones.rnn[t]:evaluate() -- make sure we are in correct testing mode
    inPut = protos.oneHot:forward(ds.input[inDex][t])
    lst = clones.rnn[t]:forward{inPut, unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
  end
  predictions = lst[#lst]
  pr('printPrictions')
  pr('1 prob')
  print(math.exp(lst[#lst][1][1]))
  pr('2 prob')
  print(math.exp(lst[#lst][1][2]))
  pr('printPrictionsEnd')
  -- print results:
  local max, idx
  max,idx = torch.max( predictions, 2)
  print(idx)
  print()
  print('Prediction:', idx[1][1], 'Label:', ds.target[inDex])
end

-- and test!
opt.test_samples = 100
for i = 1, opt.test_samples do
  local inDex = perm[i]
  print(inDex)
  test(inDex)
end
