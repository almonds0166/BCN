
import os
from pathlib import Path
import math
import itertools
from enum import Enum

import torch
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
import torchvision
from torch.utils.data import DataLoader

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Branches:
   """Base class representing branching connections.

   Each branching (indirect) connection is encoded by a matrix where each element represents the
   connection from a previous layer to the following layer. The center of this matrix represents
   the connection from a previous layer's neuron to the single nearest neighbor neuron in the next
   layer. Likewise, off center elements represent connections from a previous layer's neuron to off
   center nearest neighbor neurons.
   
   Args:
      width: Size of the matrices representing the branches, default 7.

   Attributes:
      center (int): index of the center of the matrices representing the branches.
   """
   def __init__(self, width: int=7):
      if width % 2 == 0: raise ValueError(f"Width must be odd; {width} given.")
      if width < 3: raise ValueError(f"Width must be at least 3; {width} given.")
      self.width = width
      self.center = (width - 1)//2
      self.connections = {}
      self.default = torch.zeros((width,width))
      self.default[self.center,self.center] = 1 # default is direct connections only

   def __getitem__(self, key):
      if key in self.connections: return self.connections[key]
      return self.pan(self.default, key[0], key[1])

   def __setitem__(self, key, value):
      self.connections[key] = value

   def __repr__(self):
      return f"Branches(width={self.width})"

   @staticmethod
   def pan(x, dy: int, dx: int):
      """Pan a tensor ``x`` down ``dy`` and over ``dx``.

      Similar to torch.roll, circularly convolves the given tensor, instead with zero padding.

      Args:
         x (tensor.torch): 2D matrix to pan.
         dy: Count of places to shift the tensor downward.
         dx: Count of places to shift the tensor rightward.
      """
      if dy == 0 and dx == 0: return x
      h, w = x.size()
      if abs(dy) >= h and abs(dx) >= w:
         return torch.zeros(x.size())
      # vertical
      y = torch.roll(x, dy, dims=0)
      if dy > 0:
         y[0:dy,:] = 0
      elif dy < 0:
         y[(h+dy):h,:] = 0
      # horizontal
      y = torch.roll(y, dx, dims=1)
      if dx > 0:
         y[:,0:dx] = 0
      elif dx < 0:
         y[:,(w+dx):w] = 0

      return y

class DirectOnly(Branches):
   """Branches class representing only direct connections ("pristine", "ideal").
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # there's nothing necessary to change here since the default is a direct-only kernel

   def __repr__(self):
      return "DirectOnly()"

class NearestNeighbor(Branches):
   """Branches class representing nearest neighbor connections.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      o = self.center
      for dx in (-1, 0, 1):
         for dy in (-1, 0, 1):
            self.default[o+dy,o+dx] = 1
      self.default = self.default / torch.sum(self.default)

   def __repr__(self):
      return "NearestNeighbor()"

class NearestNeighbour(Branches):
   """Alias for ``NearestNeighbor``.
   """
   def __repr__(self):
      return "NearestNeighbour()"

class NextToNN(Branches):
   """Branches class representing next-to-nearest neighbor connections.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      o = self.center
      for dx in (-2, -1, 0, 1, 2):
         for dy in (-2, -1, 0, 1, 2):
            self.default[o+dy,o+dx] = 1
      self.default = self.default / torch.sum(self.default)

   def __repr__(self):
      return "NextToNN()"

class NearestNeighborOnly(Branches):
   """Branches class representing nearest neighbor connections without the center connection.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      o = self.center
      for dx in (-1, 0, 1):
         for dy in (-1, 0, 1):
            self.default[o+dy,o+dx] = 1
      self.default[o,o] = 0
      self.default = self.default / torch.sum(self.default)

   def __repr__(self):
      return "NearestNeighborOnly()"

class NearestNeighbourOnly(Branches):
   """Alias for ``NearestNeighborOnly``.
   """
   def __repr__(self):
      return "NearestNeighbourOnly()"

class NextToNNOnly(Branches):
   """Branches class representing next-to-nearest neighbor connections without the innermost rings.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      o = self.center
      for dx in (-2, -1, 0, 1, 2):
         for dy in (-2, -1, 0, 1, 2):
            self.default[o+dy,o+dx] = 1
      for dx in (-1, 0, 1):
         for dy in (-1, 0, 1):
            self.default[o+dy,o+dx] = 0
      self.default = self.default / torch.sum(self.default)

   def __repr__(self):
      return "NextToNNOnly()"

class IndirectOnly(Branches):
   """Nearest and next-to-nearest neighbor Branches class, without the center connection.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      o = self.center
      for dx in (-2, -1, 0, 1, 2):
         for dy in (-2, -1, 0, 1, 2):
            self.default[o+dy,o+dx] = 1
      self.default[o,o] = 0
      self.default = self.default / torch.sum(self.default)

   def __repr__(self):
      return "IndirectOnly()"

class BCNLayer(nn.Module):
   """Branched connection network layer.

   Args:
      width: The side length of the layer.
      connections: The number of direct connections each neuron makes; 9, 25, 49, etc. Use
         ``float("inf")`` for a fully-connected network.
      branches (Branches): The type of indirect (branching) connections used to construct the
         branching network.
      device: The ``torch.device`` object on which the tensors will be allocated. Default is GPU if
         available, otherwise CPU.
      dropout: The proportion of dropout to use for each layer.
      mean: The mean of the normal distribution to initialize weights, default 0.0.
      std: The standard deviation of the normal distribution to initialize weights, default 0.5.
      last: Whether the layer is the final layer in the model or not, default False. If True, the
         forward output is a ``(10, _)`` tensor representing the raw, unnormalized scores of the
         ten-digit "keypad" (refer to thesis, Section _._._) ready for cross entropy loss.

   Attributes:
      Todo.
   """
   def __init__(self, width: int, connections: int, branches=DirectOnly(),
      device=DEV, dropout=0.1, mean=0.0, std=0.5, last: bool=False,
      *args, **kwargs):
      super().__init__(*args, **kwargs)
      assert connections == float("inf") \
         or math.isqrt(connections)**2 == connections, \
         (
            f"The number of connections is expected to be a perfect square "
            f"(namely 9, 25, 49, ..., float(\"inf\")); given {connections}."
         )
      # remember args
      self.height = width
      self.width = width
      self.hw = self.height*self.width
      self.connections = connections
      self.branches = branches
      ell = (math.isqrt(connections)-1)//2
      self.ells = range(-ell, ell+1) # <-- This doesn't suppert fully connected just yet! K/=;
                                     # idea to fix: make ell take the min of ell and the radius of
                                     # branches matrices
      self.device = device
      self.last = last
      # construct connection matrices
      o = self.branches.center
      self.network = {}
      for ci in self.ells:
         for cj in self.ells:
            self.network[ci,cj] = torch.zeros((self.hw,self.hw)).to(device)
            # diagonals are all as they should be, the center
            for xi in range(self.width):
               for yi in range(self.height):
                  for xo in range(self.width):
                     for yo in range(self.height):
                        # this nested loop represents the connection from
                        # source (yi,xi) to target (yo,xo)
                        dy = yo - yi
                        dx = xo - xi
                        if (o + dy < 0) or \
                           (o + dy >= self.branches.width) or \
                           (o + dx < 0) or \
                           (o + dx >= self.branches.width):
                           # skip if there's certainly no branched connection
                           continue
                        # corresponding index pair in network matrix
                        # note that Python (numpy, PyTorch) is row major
                        j = xi + self.width*yi
                        i = xo + self.width*yo
                        # put all the factors in their proper place
                        # thanks to trial and error
                        self.network[ci,cj][i,j] = self.branches[ci,cj][o+dy,o+dx]
      # initialize weights
      self.weights = {}
      for dx in self.ells:
         for dy in self.ells:
            self.weights[dy,dx] = nn.Parameter(torch.Tensor(self.hw,1)).to(device)
            nn.init.normal_(self.weights[dy,dx], mean=mean, std=std)
      # dropout
      self.dropout = nn.Dropout(p=dropout)
      # if last
      if last:
         self.mask = torch.zeros((width,width)).bool().to(device)
         i = (width-3)//2
         self.mask[i:i+3,i:i+3] = True
         self.mask[i+3,i+1] = True
         self.mask = self.mask.reshape((self.hw,1))
      else:
         self.mask = None

   def __repr__(self):
      return (
         f"BCNLayer("
         f"{self.height}x{self.width}"
         f"@{self.connections}:{self.branches}"
         f")"
      )

   def forward(self, x):
      """Forward method of this torch module.

      Args:
         x: The input tensor of size ``(features, batch_size)``.

      Returns:
         y: The output tensor. Size is ``(features, batch_size)`` if this layer is not the last
            layer, otherwise ``(10, batch_size)``.
      """
      # might there be a way to vectorize this?
      y = torch.zeros(x.size()).to(self.device)
      for dx in self.ells:
         for dy in self.ells:
            y += torch.matmul(self.network[dy,dx], x * self.weights[dy, dx])

      if self.last:
         y = self.dropout(y)
         y = torch.masked_select(y, self.mask)
      else:
         y = self.dropout(y)
         y = torch.sigmoid(y)
      
      return y

class BCN(nn.Module):
   """Branched connection network.

   Args:
      width: The side length of each layer.
      depth: The depth of the network, equal to the number of nonlinear activations.
      connections: The number of direct connections each neuron makes; 9, 25, 49, etc. Use
         ``float("inf")`` for a fully-connected network.
      branches (Branches): The type of indirect (branching) connections used to construct the
         branching networks for each layer.
      device: The ``torch.device`` object on which the tensors will be allocated. Default is GPU if
         available, otherwise CPU.
      dropout: The proportion of dropout to use for each layer.
      mean: The mean of the normal distribution to initialize weights, default 0.0.
      std: The standard deviation of the normal distribution to initialize weights, default 0.5.

   Attributes:
      Todo.
   """
   def __init__(self, width: int, depth: int, connections: int,
      branches=DirectOnly(), device=DEV, dropout: float=0.1,
      mean: float=0.0, std: float=0.5,
      *args, **kwargs):
      if depth < 1: raise ValueError(f"Depth must be at least 1; given: {depth}.")
      assert connections == float("inf") \
         or math.isqrt(connections)**2 == connections, \
         (
            f"The number of connections is expected to be a perfect square "
            f"(namely 9, 25, 49, ..., inf); given {connections}."
         )
      super().__init__(*args, **kwargs)
      # remember args
      self.height = width
      self.width = width
      self.hw = self.height*self.width
      self.depth = depth
      self.connections = connections
      self.branches = branches
      self.device = device
      # define layers
      self.layers = []
      for d in range(depth):
         self.layers.append(
            BCNLayer(
               width=width, connections=connections, branches=branches,
               device=device, dropout=dropout, mean=mean, std=std, last=(d == depth-1)
            )
         )

   def __repr__(self):
      return (
         f"BCN("
         f"{self.height}x{self.width}x{self.depth}"
         f"@{self.connections}:{self.branches}"
         f")"
      )

   def forward(self, x):
      """Forward method of this torch module.

      Args:
         x: The input tensor of size ``(features, batch_size)``.

      Returns:
         y: The output tensor of size ``(10, batch_size)``.
      """
      y = x
      for d in range(self.depth):
         y = self.layers[d](y) # sigmoid activation is applied

      return y
      return self.output(y)

class BCNDataset(Enum):
   MNIST = "MNIST"
   FASHION = "Fashion-MNIST"

def prepare_dataset(dataset=BCNDataset.MNIST, batch_size: int=32, root: str="./data/"):
   """Prepare training and validation sets of MNIST or Fashion-MNIST.
   
   Args:
      dataset (BCNDataset): The dataset to use, MNIST or FASHION.
      batch_size: The number of cookies in one batch, obviously; default 32.
      root: The directory to download the dataset to, if not already there; default "./data/".
         MNIST is downloaded to root/mnist/, and Fashion-MNIST is downloaded to root/fashion/.

   Returns:
      train: DataLoader of the training set.
      valid: DataLoader of the validation set.
   """

   assert dataset in (BCNDataset.MNIST, BCNDataset.FASHION), \
      "Given dataset must be MNIST or Fashion-MNIST."

   if dataset == BCNDataset.MNIST:
      dset = torchvision.datasets.MNIST
      path = Path(root) / "mnist/"
   elif dataset == BCNDataset.FASHION:
      dset = torchvision.datasets.FashionMNIST
      path = Path(root) / "fashion/"

   tr = torchvision.transforms.ToTensor()

   train = DataLoader(
      dset(root, download=True, train=True, transform=tr), 
      batch_size=batch_size,
      drop_last=True,
      shuffle=True
   )
   valid = DataLoader(
      dset(root, download=True, train=False, transform=tr), 
      batch_size=batch_size,
      drop_last=True,
      shuffle=True
   )

   return train, valid

if __name__ == "__main__":
   test_branches = Branches()
   test_branches[0,0] = torch.zeros(7,7); test_branches[0,0][3,3] = 4
   test_branches[1,0] = torch.zeros(7,7); test_branches[1,0][4,3] = 3
   test_branches[0,1] = torch.zeros(7,7); test_branches[0,1][3,4] = 2
   test_branches[1,1] = torch.zeros(7,7); test_branches[1,1][4,4] = 1
   model = BCN(10, 1, 9, branches=test_branches, dropout=0)

   x = torch.zeros((10,10))
   x[6,4] = 1
   print("x:"); print(x)
   x_ = x.reshape((100,1))

   A = model.layers[0].network

   print("direct test:")
   y = torch.matmul(A[0,0], x_).reshape((10,10)); print(y)

   print("down one test:")
   y = torch.matmul(A[1,0], x_).reshape((10,10)); print(y)

   print("right one test:")
   y = torch.matmul(A[0,1], x_).reshape((10,10)); print(y)

   print("diagonal test:")
   y = torch.matmul(A[1,1], x_).reshape((10,10)); print(y)
