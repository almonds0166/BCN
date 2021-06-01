
import os
from pathlib import Path
import math
import itertools
from enum import Enum
import time
import urllib.request
import json

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Branches:
   """Base class representing branching connections.

   Each branching (indirect) connection is encoded by a matrix where each element represents the
   connection from a previous layer to the following layer. The center of this matrix represents
   the connection from a previous layer's neuron to the single nearest neighbor neuron in the next
   layer. Likewise, off center elements represent connections from a previous layer's neuron to off
   center nearest neighbor neurons. Class instances act the same as Python dicts.
   
   Args:
      width: Size of the matrices representing the branches, default 9.

   Attributes:
      center (int): index of the center of the matrices representing the branches.
   """
   def __init__(self, width: int=9):
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
      return f"{self.__class__.__name__}(width={self.width})"

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
      return f"{self.__class__.__name__}()"

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
      return f"{self.__class__.__name__}()"

class NearestNeighbour(NearestNeighbor):
   """Alias for ``NearestNeighbor``.
   """
   pass

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
      return f"{self.__class__.__name__}()"

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
      return f"{self.__class__.__name__}()"

class NearestNeighbourOnly(NearestNeighborOnly):
   """Alias for ``NearestNeighborOnly``.
   """
   pass

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
      return f"{self.__class__.__name__}()"

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
      return f"{self.__class__.__name__}()"

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
      width: int=28, padding: int=0, fill: int=0,
      from_weights: str=None, save_path: str=None, **kwargs):

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
      self.from_weights = from_weights
      self.save_path = save_path

      if save_path:
         Path(save_path).mkdir(parents=True, exist_ok=True) # mkdir as needed

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
      train_times (list[float]): List of durations, in seconds, each epoch took to train.
      valid_times (list[float]): List of durations, in seconds, each epoch took to test.
      best_valid_loss (float): Minimum encountered validation loss.
      best_epoch (int): Epoch corresponding to the minimum validation loss.

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
      self.train_times = []
      self.valid_times = []
      self.best_valid_loss = float("inf")
      self.best_epoch = 0

   def __repr__(self):
      plural = self.epoch != 1
      return f"{self.__class__.__name__}({self.epoch} epoch{'s' if plural else ''})"

   def __iter__(self):
      for (k,v) in self.__dict__.items():
         yield (k,v)

   def load(self, path: str):
      """Load results from path.

      Args:
         path: File path from which to load the results.
      """
      self.__dict__ = torch.load(path)

   def save(self, path: str):
      """Save results to path.

      Args:
         path: File path to which to save the results.
      """
      torch.save(self.__dict__, path)

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

      # check if the connection matrices are already available locally under ./networks/
      fname = (
         f"{self.height}x{self.width}"
         f"@{self.connections}"
         f"-{self.branches.__class__.__name__}"
         f".pt"
      )
      fname = Path("./networks/") / fname
      if fname.exists():
         # yay!
         self.network = torch.load(fname, map_location=device)
      else:
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
         # save for later
         Path("./networks/").mkdir(exist_ok=True)
         torch.save(self.network, fname)
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
         f"{self.__class__.__name__}("
         f"{self.height}x{self.width}"
         f"@{self.connections}-{self.branches}"
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
      mean: The mean of the normal distribution to initialize weights, default 0.0.
      std: The standard deviation of the normal distribution to initialize weights, default 0.5.
      dropout (float or Tuple[float, ...]): The dropout factor to use for each layer; default 0.1.
         If provided a tuple of floats, use the values for the corresponding layer. For example,
         (0, 0.3, 0.5) will set the dropout of the third layer (and following layers if there are
         any) to 0.5, whereas the first and second layers will have dropouts of 0 and 0.3.
      verbose: Verbosity level. 0 (default) is less text, 1 is medium, 2 is most verbose.

   Attributes:
      Todo.
   """
   def __init__(self, width: int, depth: int, connections: int,
      branches=DirectOnly(), device=DEV, mean: float=0.0, std: float=0.5, dropout=0.1,
      verbose: int=0, *args, **kwargs):
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
      # set up training scheme and results attributes
      self.scheme = None
      self.results = Results()
      # define layers
      if isinstance(dropout, (int, float)):
         dropout = (dropout,) # convert to tuple
      self.layers = nn.ModuleList()
      for d in range(depth):
         self.layers.append(
            BCNLayer(
               width=width, connections=connections, branches=branches, device=device,
               dropout=dropout[min(len(dropout)-1,d)], mean=mean, std=std, last=(d == depth-1)
            )
         )

   def __repr__(self):
      return (
         f"{self.__class__.__name__}("
         f"{self.height}x{self.width}x{self.depth}"
         f"@{self.connections}-{self.branches}"
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

   def train(self, scheme, from_weights: str=None, save_path: str=None, trial: int=None):
      """Set the model to training mode and update the training scheme.

      Args:
         scheme: The training scheme that this model should follow.
         from_weights: Weights file to begin training from; default is ``None``, to initialize
            weights randomly.
         save_path: Directory to save weights under; default is None, not to save weights.
         trial: Assign the model a trial number, for the sake of repeating experiments. Default is
            None, in which case the model isn't assigned a trial number.
      """
      if self.verbose: print("Setting training scheme...")
      super().train()
      self.scheme = scheme
      self.loss_fn = nn.CrossEntropyLoss()
      self.optim = scheme.optim(self.parameters(), **scheme.optim_params)
      self.trial = trial
      # load weights if there are any given to load
      if from_weights:
         self.load_state_dict(torch.load(from_weights))

   def run_epoch(self):
      """Train for one epoch.
      """
      assert self.scheme is not None, \
         "Before training, please explicitly set this model to training mode with .train(scheme)."

      # train
      stopwatch = time.time()
      train_loss = 0
      pbar = tqdm(self.scheme.train, desc=f"Epoch {self.results.epoch}", unit="b")
      for i, (batch, labels) in enumerate(pbar):
         # model expects batch_size as last dimension
         batch = torch.transpose(batch, 0, 1).to(self.device)
         labels = labels.to(self.device)
         self.optim.zero_grad()
         predictions = torch.roll(self.forward(batch), -1, 1) # keypad fix, see Section _._._
         loss = self.loss_fn(predictions, labels)
         train_loss += loss.item()
         pbar.set_postfix(loss=f"{loss.item():.2f}")
         loss.backward()
         self.optim.step()
      # average loss
      train_loss /= len(self.scheme.train)
      # record training loss
      self.results.train_losses.append(train_loss)
      if self.verbose:
         print(f"train_loss: {train_loss} (average)")

      self.results.train_times.append(time.time() - stopwatch)

      # validation loss
      stopwatch = time.time()
      valid_loss = 0
      correct = 0
      precision = 0
      recall = 0
      f1_score = 0
      with torch.no_grad():
         for i, (batch, labels) in enumerate(self.scheme.valid):
            # model expects batch_size as last dimension
            batch = torch.transpose(batch, 0, 1).to(self.device)
            labels = labels.to(self.device)
            predictions = torch.roll(self.forward(batch), -1, 1)
            pred = torch.argmax(predictions, dim=1)
            # loss
            loss = self.loss_fn(predictions, labels)
            valid_loss += loss.item()
            # accuracy
            correct += sum(pred == labels)
            # precision, recall, f1 score
            p, r, f1, _ = precision_recall_fscore_support(
               labels.cpu(), pred.cpu(), average="weighted", zero_division=0)
            precision += p
            recall += r
            f1_score += f1
      # average the metrics
      N = len(self.scheme.valid)
      valid_loss = valid_loss / N
      accuracy   = correct.item() / (N*self.scheme.batch_size)
      precision  = precision / N
      recall     = recall / N
      f1_score   = f1_score / N
      # record metrics
      self.results.valid_losses.append(valid_loss)
      self.results.accuracies.append(accuracy)
      self.results.precisions.append(precision)
      self.results.recalls.append(recall)
      self.results.f1_scores.append(f1_score)

      if self.verbose:
         print(f"valid_loss: {valid_loss} (average)")

      if valid_loss < self.results.best_valid_loss:
         self.results.best_valid_loss = valid_loss
         self.results.best_epoch = self.results.epoch
         if self.verbose:
            print("Model improved!")

         # save weights if path was provided
         if self.scheme.save_path:
            trial = "" if self.trial is None else f".t{self.trial}"
            fname = (
               f"weights"
               f"_{self.height}x{self.width}x{self.depth}"
               f"@{self.connections}"
               f"-{self.branches.__class__.__name__}"
               f".b{self.scheme.batch_size}"
               f"{trial}"
               f".pt"
            )
            fname = Path(self.scheme.save_path) / fname
            torch.save(self.state_dict(), fname)
            if self.verbose >= 2:
               print(f"Saved weights to: {fname}")

      self.results.valid_times.append(time.time() - stopwatch)
      self.results.epoch += 1

      # update results file if path was provided
      if self.scheme.save_path:
         trial = "" if self.trial is None else f".t{self.trial}"
         fname = (
            f"results"
            f"_{self.height}x{self.width}x{self.depth}"
            f"@{self.connections}"
            f"-{self.branches.__class__.__name__}"
            f".b{self.scheme.batch_size}"
            f"{trial}"
            f".pkl"
         )
         self.results.save(fname)

   def run_epochs(self, n: int, webhook: str=None):
      """Train for n epochs.

      Args:
         n: The number of epochs to train for.
         webhook: The Discord or Slack webhook URL to post to. See `here`_ for what it looks like.
         
      .. _here: https://i.imgur.com/Z8qiTE2.png
      """
      if n <= 0: return

      for e in range(n):
         self.run_epoch()

      if webhook:
         total_time = round(sum(self.results.train_times) + sum(self.results.valid_times))
         epochs = f"{n} epoch" + ("s" if n != 1 else "")
         content = (
            f"Finished training `{repr(self)}` for {epochs}! "
               f"(took around {total_time} seconds total)\n"
            f"The epoch with best performance was epoch {self.results.best_epoch}:\n"
            f"* Validation loss: {self.results.best_valid_loss}\n"
            f"* F1 score: {self.results.f1_scores[self.results.best_epoch]}\n"
         )
         if "hooks.slack.com" in webhook:
            payload = {"text": content} # slack
         else:
            payload = {"content": content} # discord
         data = json.dumps(payload).encode("utf-8")
         req = urllib.request.Request(webhook)
         req.add_header("Content-Type", "application/json; charset=utf-8")
         req.add_header("User-Agent", "Almonds/0.0")
         req.add_header("Content-Length", len(data))
         response = urllib.request.urlopen(req, data)

if __name__ == "__main__":
   torch.manual_seed(23)
   num_epochs = 1
   # prepare model
   model = BCN(30, 3, 9, dropout=0.1, verbose=1)
   # prepare for training
   scheme = TrainingScheme(width=30, padding=1, batch_size=64)
   model.train(scheme)
   # train
   model.run_epochs(num_epochs)
   # results
   print("Results:")
   for key, value in model.results:
      print(f"\t{key}: {value}")