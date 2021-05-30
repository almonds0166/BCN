
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
from tqdm import tqdm

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

class Dataset(Enum):
   MNIST = "MNIST"
   FASHION = "Fashion-MNIST"

class TrainingScheme:
   """Class representing how a BCN model should be trained, including dataset and hyperparameters.

   Args:
      optim: The torch optim instance that should be used for training. Default is `Adam with
         weight decay`_.
      dataset (Dataset): The dataset to use, MNIST or FASHION.
      batch_size: The number of cookies in one batch, obviously; default 32.
      root: The directory to download the dataset to, if not already there; default "./data/".
      width: The desired final height and width of the dataset images; default 28. Note that the
         images are vectorized, so a width of 28 ultimately corresponds to a batch of size of
         (batch_size, 784) instead of (batch_size, 28, 28).
      padding: The number of rings of the padding value to add around the outside of each dataset
         image. Default value is 0, to add no padding. Note: Padding is added after the resizing
         transformation such that the final image size is (width, width).
      fill: The value to pad with, if padding; default 0.
      **kwargs: The extra keyword arguments are passed into the optimizer (for e.g. lr, eps, ...).

   Attributes:
      optim_params: The keyword arguments passed into the optimizer.

   .. _Adam with weight decay: https://www.fast.ai/2018/07/02/adam-weight-decay/
   """
   def __init__(self, optim=None, dataset=Dataset.MNIST, batch_size: int=32, root: str="./data/",
      width: int=28, padding: int=0, fill: int=0, **kwargs):

      assert dataset in (Dataset.MNIST, Dataset.FASHION), \
         "Given dataset must be MNIST or Fashion-MNIST."
      
      self.optim = optim if optim is not None else torch.optim.AdamW
      self.optim_params = kwargs

      self.dataset = dataset
      self.batch_size = batch_size
      self.root = root
      self.width = width
      self.padding = padding
      self.fill = fill

      self.prepare_dataset()

   def prepare_dataset(self):
      """Prepare the training and validation sets of MNIST or Fashion-MNIST.
      """
      if self.dataset == Dataset.MNIST:
         dset = torchvision.datasets.MNIST
      elif self.dataset == Dataset.FASHION:
         dset = torchvision.datasets.FashionMNIST

      # transformations
      tr = torchvision.transforms.Compose([
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Resize((self.width-2*self.padding)),
         torchvision.transforms.Pad(self.padding, fill=self.fill),
         lambda image: torch.reshape(image, (image.size().numel(),)) # vectorize!
      ])

      train = DataLoader(
         dset(self.root, download=True, train=True, transform=tr), 
         batch_size=self.batch_size,
         drop_last=True,
         shuffle=True,
         pin_memory=torch.cuda.is_available()
      )
      valid = DataLoader(
         dset(self.root, download=True, train=False, transform=tr), 
         batch_size=self.batch_size,
         drop_last=True,
         shuffle=True,
         pin_memory=torch.cuda.is_available()
      )

      self.train = train
      self.valid = valid

      return train, valid

class Results:
   """Class representing BCN training results.

   Attributes:
      epoch (int): The number of epochs the BCN model has been trained for.
      train_losses (list[float]): List of average training set losses acorss epochs.
      valid_losses (list[float]): List of average validation set losses across epochs.
      accuracies (list[float]): List of validation set accuracies across epochs.
      precisions (list[float]): List of validation set `precision scores`_ across eopchs.
      recalls (list[float]): List of validation set `recall scores`_ across epochs.
      f1_scores (list[float]): List of validation set `F1 scores`_ across epochs.

   .. _precision scores: https://en.wikipedia.org/wiki/Precision_and_recall
   .. _recall scores: https://en.wikipedia.org/wiki/Precision_and_recall
   .. _F1 scores: https://en.wikipedia.org/wiki/F-score
   """
   def __init__(self):
      self.epoch = 0
      self.train_losses = []
      self.valid_losses = []
      self.accuracies = []
      self.precisions = []
      self.recalls = []
      self.f1_scores = []

   def __repr__(self):
      return 

class BCNLayer(nn.Module):
   """Branched connection network layer.

   Args:
      width: The side length of the layer.
      connections: The number of direct connections each neuron makes; 9, 25, 49, etc. Use
         ``None`` for a fully-connected network.
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
   def __init__(self, width: int, connections: int=None, branches=DirectOnly(),
      device=DEV, dropout=0.1, mean=0.0, std=0.5, last: bool=False,
      *args, **kwargs):
      super().__init__(*args, **kwargs)
      assert connections is None \
         or int(math.sqrt(connections))**2 == connections, \
         (
            f"The number of connections is expected to be a perfect square "
            f"(namely 9, 25, 49, ...) of None; given {connections}."
         )
      # remember args
      self.height = width
      self.width = width
      self.hw = self.height*self.width
      self.connections = connections
      self.branches = branches
      ell = (branches.width-1)//2 if connections is None else (int(math.sqrt(connections))-1)//2
      self.ells = range(-ell, ell+1)
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
            self.weights[dy,dx] = nn.Parameter(torch.Tensor(self.hw,1).to(device))
            nn.init.normal_(self.weights[dy,dx], mean=mean, std=std)
            self.register_parameter(f"({dy},{dx})", self.weights[dy,dx])
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
         batch_size = y.size()[-1]
         y = self.dropout(y)
         y = torch.masked_select(y, self.mask)
         y = y.reshape((10,batch_size))
         y = torch.transpose(y, 0, 1) # CrossEntropyLoss has batch first
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
         ``None`` for a fully-connected network.
      branches (Branches): The type of indirect (branching) connections used to construct the
         branching networks for each layer.
      device: The ``torch.device`` object on which the tensors will be allocated. Default is GPU if
         available, otherwise CPU.
      dropout: The proportion of dropout to use for each layer.
      mean: The mean of the normal distribution to initialize weights, default 0.0.
      std: The standard deviation of the normal distribution to initialize weights, default 0.5.
      verbose: Verbosity level. 0 (default) is less text, 1 is medium, 2 is most verbose.
      tqdm: The tqdm wrapper to use (use tqdm.notebook.tqdm if in a notebook).

   Attributes:
      Todo.
   """
   def __init__(self, width: int, depth: int, connections: int,
      branches=DirectOnly(), device=DEV, dropout: float=0.1,
      mean: float=0.0, std: float=0.5, verbose: int=0, tqdm=tqdm,
      *args, **kwargs):
      if depth < 1: raise ValueError(f"Depth must be at least 1; given: {depth}.")
      assert connections is None \
         or int(math.sqrt(connections))**2 == connections, \
         (
            f"The number of connections is expected to be a perfect square "
            f"(namely 9, 25, 49, ...) or None; given {connections}."
         )
      super().__init__(*args, **kwargs)
      # remember args
      self.height = width
      self.width = width
      self.hw = self.height*self.width
      self.depth = depth
      self.connections = connections
      self.branches = branches
      if verbose: print(f"Building BCN model {self.__repr__()}...")
      self.device = device
      self.verbose = verbose
      self.tqdm = tqdm
      # set up training scheme and results attributes
      self.scheme = None
      self.results = Results()
      # define layers
      self.layers = nn.ModuleList()
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
         x: The input tensor of size (features, batch_size).

      Returns:
         y: The output tensor of size (10, batch_size).
      """
      y = x
      for d in range(self.depth):
         y = self.layers[d](y) # sigmoid activation is applied

      return y

   def train(self, scheme):
      """Set the model to training mode and update the training scheme.

      Args:
         scheme: The training scheme that this model should follow.
      """
      super().train()
      self.scheme = scheme
      self.loss_fn = nn.CrossEntropyLoss()
      self.optim = scheme.optim(self.parameters(), **scheme.optim_params)

   def run_epoch(self):
      """Train for one epoch.
      """
      assert self.scheme is not None, \
         "Before training, please explicitly set this model to training mode with .train(scheme)."

      # train
      train_loss = 0
      pbar = self.tqdm(self.scheme.train, desc=f"Epoch {self.results.epoch}", unit="b")
      for i, (batch, labels) in enumerate(pbar):
         # model expects batch_size as last dimension
         batch = torch.transpose(batch, 0, 1)
         self.optim.zero_grad()
         predictions = model.forward(batch)
         loss = self.loss_fn(predictions, labels)
         train_loss += loss.item()
         pbar.set_postfix(loss=f"{loss.item():.2f}")
         loss.backward()
         self.optim.step()
      train_loss /= len(self.scheme.train)
      self.results.train_losses.append(train_loss)
      if self.verbose:
         print(f"train_loss: {train_loss} (average)")

      # validation loss
      valid_loss = 0
      with torch.no_grad():
         for i, (batch, labels) in enumerate(self.scheme.valid):
            # model expects batch_size as last dimension
            batch = torch.transpose(batch, 0, 1)
            predictions = model.forward(batch)
            loss = self.loss_fn(predictions, labels)
            valid_loss += loss.item()
      valid_loss /= len(self.scheme.valid)
      self.results.valid_losses.append(valid_loss)
      if self.verbose:
         print(f"valid_loss: {valid_loss} (average)")

      self.results.epoch += 1

   def run_epochs(self, n: int):
      """Train for n epochs.

      Args:
         n: The number of epochs to train for.
      """
      for e in range(n):
         self.run_epoch()

if __name__ == "__main__":
   torch.manual_seed(23)
   num_epochs = 5
   # prepare model
   model = BCN(30, 2, 9, dropout=0.1, verbose=1)
   # prepare for training
   scheme = TrainingScheme(width=30, padding=1, batch_size=64)
   model.train(scheme)
   # train
   model.run_epochs(num_epochs)
   # results
   print(f"train_losses: {model.results.train_losses}")
   print(f"valid_losses: {model.results.valid_losses}")