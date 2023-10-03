import torch
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
import operations
from torch.autograd import Variable
import networkx as nx
from networkx.readwrite import json_graph
import json
import time

import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import nni.retiarii.strategy as strategy


from nni.nas.nn.pytorch.mutation_utils import Mutable, generate_new_label, get_fixed_value


@model_wrapper
class MixedOp(nn.Module):

  def __init__(self, C, stride, primitives=None, op_dict=None, weighting_algorithm=None, label=None):
    """ Perform a mixed forward pass incorporating multiple primitive operations like conv, max pool, etc.

    # Arguments

      primitives: the list of strings defining the operations to choose from.
      op_dict: The dictionary of possible operation creation functions.
        All primitives must be in the op dict.
    """

    self._label = generate_new_label(label)
    super().__init__()
    self._ops = []
    self._stride = stride
    if primitives is None:
          primitives = PRIMITIVES
    self._primitives = primitives
    if op_dict is None:
          op_dict = operations.OPS
    for primitive in primitives:
      op = op_dict[primitive](C, C, stride, False)
      # op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)
    
    self.op = nn.LayerChoice(self._ops, label=f'{self.label}')

  @property
  def label(self):
      return self._label
  
  def get_search_space(self):
    return self._primitives

  def forward(self, x):
    # result = 0
    # print('-------------------- forward')
    # print('weights shape: ' + str(len(weights)) + ' ops shape: ' + str(len(self._ops)))
    # for i, (w, op) in enumerate(zip(weights, self._ops)):
    #   print('w shape: ' + str(w.shape) + ' op type: ' + str(type(op)) + ' i: ' + str(i) + ' self._primitives[i]: ' + str(self._primitives[i]) + 'x size: ' + str(x.size()) + ' stride: ' + str(self._stride))
    #   op_out = op(x)
    #   print('op_out size: ' + str(op_out.size()))
    #   result += w * op_out
    # return result
    # apply all ops with intensity corresponding to their weight

    # return sum(w * op(x) for w, op in zip(weights, self._ops))

    # max_w = torch.max(weights)
    # return sum((1. - max_w + w) * op(x) for w, op in zip(weights, self._ops))
    return self.op(x)




@model_wrapper
class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, primitives=None, op_dict=None, weighting_algorithm=None, label=None):
    """Create a searchable cell representing multiple architectures.

    The Cell class in model.py is the equivalent for a single architecture.

    # Arguments
      steps: The number of primitive operations in the cell,
        essentially the number of low level layers.
      multiplier: The rate at which the number of channels increases.
      op_dict: The dictionary of possible operation creation functions.
        All primitives must be in the op dict.
    """
    super().__init__()
    self._label = generate_new_label(label)
    self.reduction = reduction

    if reduction_prev is None:
      self.preprocess0 = operations.Identity()
    elif reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, stride=2, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, primitives, op_dict, weighting_algorithm=weighting_algorithm, label=f'{self.label}/op_{i+2}_{j}')
        self._ops.append(op)


    self.input_switch = nn.ModuleList()
    for i in range(self._steps):
      self.input_switch.append(nn.InputChoice(n_candidates=i+2, n_chosen=min(2, i+1),
                                              reduction='sum', label=f'{self.label}/input_{i+2}'))


  @property
  def label(self):
      return self._label
  

  def get_search_space(self):
    ss = {}
    states_cnt = 2
    offset = 0
    for i in range(self._steps):
      ss[f'{self.label}/input_{i+2}'] = [self._ops[offset+j].label for j in range(states_cnt)]

      offset += states_cnt
      states_cnt += 1 
    return ss
      

  def forward(self, s0, s1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = self.input_switch[i]([self._ops[offset+j](h) for j, h in enumerate(states)])
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)



@model_wrapper
class Network(nn.Module):

  def __init__(self, C=16, num_classes=10, layers=8, criterion=None, steps=4, multiplier=4, stem_multiplier=3,
               in_channels=3, primitives=None, op_dict=None, C_mid=None, weights_are_parameters=False,
               weighting_algorithm=None):
    super().__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    if criterion is None:
      self._criterion = nn.CrossEntropyLoss()
    else:
      self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._weights_are_parameters = weights_are_parameters
    if primitives is None:
      primitives = PRIMITIVES
    self.primitives = primitives

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False

    self._normal_arch = {}
    self._reduce_arch = {}

    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                  reduction, reduction_prev, primitives, op_dict,
                  weighting_algorithm=weighting_algorithm, label='reduce' if reduction else 'normal')
      if reduction:
        self._reduce_arch = cell.get_search_space()
      else:
        self._normal_arch = cell.get_search_space()
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)


    # self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def get_search_space(self):
    ss = {
      'normal': self._normal_arch,
      'reduce': self._reduce_arch,
    }

    return ss


  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  # def _initialize_alphas(self):
  #   k = sum(1 for i in range(self._steps) for n in range(2+i))
  #   num_ops = len(self.primitives)

  #   self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
  #   self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
  #   if self._weights_are_parameters:
  #     # in simpler training modes the weights are just regular parameters
  #     self.alphas_normal = torch.nn.Parameter(self.alphas_normal)
  #     self.alphas_reduce = torch.nn.Parameter(self.alphas_reduce)
  #   self._arch_parameters = [
  #     self.alphas_normal,
  #     self.alphas_reduce,
  #   ]

  # def arch_parameters(self):
  #   return self._arch_parameters

  # def arch_weights(self, stride_idx):
  #   weights_softmax_view = self._arch_parameters[stride_idx]
  #   # apply softmax and convert to an indexable view
  #   weights = F.softmax(weights_softmax_view, dim=-1)
  #   return weights

  # def genotype(self, skip_primitive='none'):
  #   '''
  #   Extract the genotype, or specific connections within a cell, as encoded by the weights.
  #   # Arguments
  #       skip_primitives: hack was added by DARTS to temporarily workaround the
  #           'strong gradient' problem identified in the sharpDARTS paper https://arxiv.org/abs/1903.09900,
  #           set skip_primitive=None to not skip any primitives.
  #   '''
  #   gene_normal = genotype_extractor.parse_cell(
  #     F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),
  #     primitives=self.primitives, steps=self._steps, skip_primitive=skip_primitive)
  #   gene_reduce = genotype_extractor.parse_cell(
  #     F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),
  #     primitives=self.primitives, steps=self._steps, skip_primitive=skip_primitive)

  #   concat = range(2+self._steps-self._multiplier, self._steps+2)
  #   genotype = Genotype(
  #     normal=gene_normal, normal_concat=concat,
  #     reduce=gene_reduce, reduce_concat=concat,
  #     layout='cell',
  #   )
  #   return genotype