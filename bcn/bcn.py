
import os
from pathlib import Path
import math
import itertools
from enum import Enum
import time
import urllib.request
import json
from typing import Any, Union, Optional, Tuple, List, Set, Sequence, Dict

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from branches import Branches
from branches.simple import DirectOnly

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Connections(Enum):
   """Enum class representing the number of directed connections AKA "arms".
   """
   ONE_TO_9 = 9
   ONE_TO_25 = 25
   ONE_TO_49 = 49
   ONE_TO_81 = 81
   FULLY_CONNECTED = "Inf" # use with care

class Dataset(Enum):
   """Enum class representing datasets, MNIST and Fashion-MNIST.
   """
   MNIST = "MNIST"
   FASHION = "Fashion-MNIST"

class TrainingScheme:
   """Class representing how a BCN model should be trained, including dataset and hyperparameters.

   Args:
      optim: The torch Optimizer class that should be used for training. Default is `Adam with
         weight decay`_, AKA ``torch.optim.AdamW``.
      dataset: The dataset to use, MNIST or FASHION.
      batch_size: The batch size used for training, default 32.
      dataset_path: The directory to download the dataset to, if not already there; default is
         ``./data/``.
      width: The desired final height and width of the dataset images; default 28. Note that the
         images are vectorized, so a width of 28 ultimately corresponds to a batch of size of
         (``batch_size``, 784) instead of (``batch_size``, 28, 28).
      padding: The number of rings of the padding value to add around the outside of each dataset
         image. Default value is 0, to add no padding. Note: Padding is added after the resizing
         transformation so that the final image size is (``width``, ``width``).
      fill: The value to pad with, if padding; default 0.
      **kwargs: The extra keyword arguments are passed into the optimizer (for e.g. ``lr``, ...).

   Attributes:
      optim (type): The torch Optimizer used for training.
      dataset (Dataset): The dataset used, MNIST or FASHION.
      batch_size (int): The batch size used for training.
      dataset_path (Union[Path,str]): The directory to download the dataset to, if not already
         there.
      width (int): The height and width of the dataset images. Note that the images are vectorized,
         so a width of 28 ultimately corresponds to a batch of size of (``batch_size``, 784)
         instead of (``batch_size``, 28, 28).
      padding (int): The number of rings of the padding value to add around the outside of each
         dataset image. Note: Padding is added after the resizing transformation so that the final
         image size is (``width``, ``width``) before vectorization.
      fill (int): The value to pad with, if padding.
      optim_params (Dict[str,Any]): The keyword arguments passed to the instance, to be use by the
         optimizer.
      train (DataLoader): Torch DataLoader representing the training set.
      valid (DataLoader): Torch DataLoader representing the validation set.

   .. _Adam with weight decay: https://www.fast.ai/2018/07/02/adam-weight-decay/
   """
   def __init__(self, *,
      optim: Optional[type]=None,
      dataset: Dataset=Dataset.MNIST,
      batch_size: int=32,
      dataset_path: Union[Path,str]=Path("./data/"),
      width: int=28,
      padding: int=0,
      fill: int=0,
      **kwargs
   ):

      assert dataset in (Dataset.MNIST, Dataset.FASHION), \
         "Given dataset must be MNIST or Fashion-MNIST."
      
      self.optim = optim if optim is not None else torch.optim.AdamW
      self.optim_params = kwargs
      self.dataset = dataset
      self.batch_size = batch_size
      self.dataset_path = dataset_path
      self.width = width
      self.padding = padding
      self.fill = fill

      self.prepare_dataset() # load self.train and self.valid

   def prepare_dataset(self) -> Tuple[DataLoader,DataLoader]:
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
         dset(self.dataset_path, download=True, train=True, transform=tr), 
         batch_size=self.batch_size,
         drop_last=True,
         shuffle=True,
         pin_memory=torch.cuda.is_available()
      )
      valid = DataLoader(
         dset(self.dataset_path, download=True, train=False, transform=tr), 
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
      train_losses (List[float]): List of average training set losses acorss epochs.
      valid_losses (List[float]): List of average validation set losses across epochs.
      accuracies (List[float]): List of validation set accuracies across epochs.
      precisions (List[float]): List of validation set `precision scores`_ across eopchs.
      recalls (List[float]): List of validation set `recall scores`_ across epochs.
      f1_scores (List[float]): List of validation set `F1 scores`_ across epochs.
      train_times (List[float]): List of durations, in seconds, each epoch took to train.
      valid_times (List[float]): List of durations, in seconds, each epoch took to test.
      best_valid_loss (float): Minimum encountered validation loss.
      best_epoch (int): Epoch corresponding to the minimum validation loss.
      tag (str): Anything notable about the model or results. Used as plot titles when plotting.
         Set via the BCN.train method.

   .. _precision scores: https://en.wikipedia.org/wiki/Precision_and_recall
   .. _recall scores: https://en.wikipedia.org/wiki/Precision_and_recall
   .. _F1 scores: https://en.wikipedia.org/wiki/F-score
   """
   def __init__(self, tag: str=""):
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
      self.tag = ""

   def __repr__(self):
      plural = self.epoch != 1
      return f"{self.__class__.__name__}({self.epoch} epoch{'s' if plural else ''})"

   def __iter__(self):
      for (k,v) in self.__dict__.items():
         yield (k,v)

   def load(self, path: Union[Path,str]) -> None:
      """Load results from path.

      Args:
         path: File path from which to load the results.
      """
      self.__dict__ = torch.load(path)

   def save(self, path: Union[Path,str]) -> None:
      """Save results to path.

      Args:
         path: File path to which to save the results.
      """
      torch.save(self.__dict__, path)

class BCNLayer(nn.Module):
   """Represents a branched connection network layer.

   Args:
      width: The side length of the layer.
      connections: The number of direct connections each neuron makes. Default is 1-to-9.
      branches: The type of indirect (branching) connections used to construct the branching
         network. Default is direct connections only.
      device: The ``torch.device`` object on which the tensors will be allocated. Default is GPU if
         available, otherwise CPU.
      dropout: The proportion of dropout to use for this layer, default 0.1.
      mean: The mean of the normal distribution to initialize weights, default 0.0.
      std: The standard deviation of the normal distribution to initialize weights, default 0.5.
      last: Whether the layer is the final layer in the model or not, default False. If True, the
         forward output is a (10, -1) tensor representing the raw, unnormalized scores of the
         ten-digit "keypad" (refer to thesis, Section _._._) ready for cross entropy loss.

   Attributes:
      width (int): The side length of the layer.
      hw (int): The product of the layer's height and width, namely ``width * width`` in this
         version of BCN.
      connections (Connections): The number of direct connections each neuron makes.
      branches (Branches): The type of indirect (branching) connections used to construct the
         branching network.
      device (torch.device): The ``torch.device`` object on which the tensors will be allocated.
         Default is GPU if available, otherwise CPU.
      dropout (nn.Dropout): The torch Dropout module use when training.
      mean (float): The mean of the normal distribution used to initialize weights.
      std (float): The standard deviation of the normal distribution used to initialize weights.
      last (bool): Whether the layer is the final layer in the model or not. If ``True``, the
         forward output is a (10, -1) tensor representing the raw, unnormalized scores of the
         ten-digit "keypad" (refer to thesis, Section _._._) ready for cross entropy loss.
      ells (range): A range of offsets, centered around 0, used for the direct connections. For
         example, 1-to-25 connections will range from -2 to +2 inclusive, because this represents
         a span of width 5.
      network (Dict[Tuple[int,int],torch.Tensor]): In future versions, this will probably be a
         tensor for performance reasons. I'll hold off on complete documentation for now.
      weights (Dict[Tuple[int,int],nn.Parameter]): In future versions, this will probably be a
         tensor for performance reasons. I'll hold off on complete documentation for now.
      mask (Optional[torch.Tensor]): If this is a last layer, the mask attribute represents a
         tensor that filters the output to ten values. ``None`` if this is not a last layer.
   """
   def __init__(self, width: int, *,
      connections: Connections=Connections.ONE_TO_9,
      branches: Branches=DirectOnly(),
      device: torch.device=DEV,
      dropout: float=0.1,
      mean: float=0.0,
      std: float=0.5,
      last: bool=False
   ):
      super().__init__()
      # remember args
      self.height = width
      self.width = width
      self.hw = self.height*self.width
      self.connections = connections
      self.branches = branches
      if connections == Connections.FULLY_CONNECTED:
         ell = (branches.width-1)//2
      else:
         ell = (int(math.sqrt(connections.value))-1)//2
      self.ells = range(-ell, ell+1)
      self.device = device
      self.last = last

      # check if the connection matrices are already available locally under ./networks/
      fname = (
         f"{self.height}x{self.width}"
         f"@{self.connections.value}"
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
         f"@{self.connections.value}-{self.branches}"
         f")"
      )

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      """The forward computation performed at every BCNLayer call.

      Note:
         Call the BCNLayer instance instead of using this method directly.

      Args:
         x: The input tensor of size (``features``, ``batch_size``).

      Returns:
         y: The output tensor. Size is (``features``, ``batch_size``) if this layer is not the last
            layer, otherwise (10, ``batch_size``).
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
   """Represents a branched connection network.

   Args:
      width: The side length of each layer.
      depth: The depth of the network, equal to the number of nonlinear activations.
      connections: The number of direct connections each neuron makes. Default is 1-to-9.
      branches: The type of indirect (branching) connections used to construct the branching
         networks for each layer. Default is direct connections only.
      device: The ``torch.device`` object on which the tensors will be allocated. Default is GPU if
         available, otherwise CPU.
      mean: The mean of the normal distribution to initialize weights, default 0.0.
      std: The standard deviation of the normal distribution to initialize weights, default 0.5.
      dropout: The dropout factor to use for each layer; default 0.1. If provided a tuple of
         floats, use the values for the corresponding layer. For example, (0, 0.3, 0.5) will set
         the dropout of the third layer (and following layers if there are any) to 0.5, whereas the
         first and second layers will have dropouts of 0 and 0.3 respectively.
      verbose: Verbosity level. 0 (default) is no text, 1 is some, 2 is most verbose. Might become
         deprecated in future versions.

   Attributes:
      width (int): The side length of each layer.
      hw (int): The product of each layer's height and width, namely ``width * width`` in this
         version of BCN.
      depth (int): The depth of the network, equal to the number of nonlinear activations.
      connections (Connections): The number of direct connections each neuron makes.
      branches (Branches): The type of indirect (branching) connections used to construct the
         branching networks for each layer. Default is direct connections only.
      device (torch.device): The ``torch.device`` object on which the tensors will be allocated.
      mean (float): The mean of the normal distribution used to initialize weights.
      std (float): The standard deviation of the normal distribution used to initialize weights.
      dropout (Tuple[float,...]): The proportion of dropout to use for each layer, as a tuple of
         floats corresponding to the first layer, second, and so on. If the length of this tuple is
         less than the number of layers, then the reamining layers use the last value in the tuple.
      verbose (int): Verbosity level. 0 (default) is no text, 1 is some, 2 is most verbose. Might
         become deprecated in future versions.
      trial (Optional[int]): The trial of this model experiment, specified by the BCN.train method.
         Used when naming the weights & results files. If ``None``, this model does not represent
         any particular trial.
      scheme (Optional[TrainingScheme]): The training scheme to use when training this model.
         Specified by the BCN.train method.
      results (Results): The model training results.
      layers (nn.ModuleList): The list of BCNLayer layers.
   """
   def __init__(self, width: int, depth: int, *,
      connections: Connections=Connections.ONE_TO_9,
      branches: Branches=DirectOnly(),
      device: torch.device=DEV,
      mean: float=0.0,
      std: float=0.5,
      dropout: Union[Tuple[float,...],float]=0.1,
      verbose: int=0,
      **kwargs
   ):
      if depth < 1: raise ValueError(f"Depth must be at least 1; given: {depth}.")
      super().__init__(*args, **kwargs)
      # remember args
      self.height = width
      self.width = width
      self.hw = self.height*self.width
      self.depth = depth
      self.connections = connections
      self.branches = branches
      self.save_path = None
      self.trial = None
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
         f"@{self.connections.value}-{self.branches}"
         f")"
      )

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      """The forward computation performed at every BCN call.

      Note:
         Call the BCN instance instead of using this method directly.

      Args:
         x: The input tensor of size (``features``, ``batch_size``).

      Returns:
         y: The output tensor of size (10, ``batch_size``).
      """
      y = x
      for d in range(self.depth):
         y = self.layers[d](y) # sigmoid activation is applied, except at end

      return y

   def train(self, scheme, *,
      from_weights: Union[Path,str,None]=None,
      save_path: Union[Path,str,None]=None,
      trial: Optional[int]=None,
      tag: str=""
   ) -> None:
      """Set the model to training mode and update the training scheme.

      Sets the training scheme for this model, and switches the model to training mode. Loads
      weights if given. Also specifies some model attributes related to training, as given.

      Args:
         scheme: The training scheme that this model should follow.
         from_weights: Weights file to begin training from; default is ``None``, to initialize
            weights randomly.
         trial: Assign the model a trial number, for the sake of repeating experiments. Default is
            ``None``, in which case the model isn't assigned a trial number.
         tag: Anything notable about the model or results. Used as plot titles when plotting.
      """
      if self.verbose: print("Setting training scheme...")
      super().train()
      self.scheme = scheme
      self.loss_fn = nn.CrossEntropyLoss()
      self.optim = scheme.optim(self.parameters(), **scheme.optim_params)
      self.trial = trial
      self.save_path = save_path
      # load weights if there are any given to load
      if from_weights:
         self.load_state_dict(torch.load(from_weights))
      if save_path:
         Path(save_path).mkdir(parents=True, exist_ok=True) # mkdir as needed

   def run_epoch(self) -> None:
      """Train for one epoch.

      Note:
         Remember to set the training scheme before running this method.
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
         predictions = torch.roll(self(batch), -1, 1) # keypad fix, see Section _._._
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
            predictions = torch.roll(self(batch), -1, 1)
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
         if self.save_path:
            trial = "" if self.trial is None else f".t{self.trial}"
            fname = (
               f"weights"
               f"_{self.height}x{self.width}x{self.depth}"
               f"@{self.connections.value}"
               f"-{self.branches.__class__.__name__}"
               f".b{self.scheme.batch_size}"
               f"{trial}"
               f".pt"
            )
            fname = Path(self.save_path) / fname
            torch.save(self.state_dict(), fname)
            if self.verbose >= 2:
               print(f"Saved weights to: {fname}")

      self.results.valid_times.append(time.time() - stopwatch)
      self.results.epoch += 1

      # update results file if path was provided
      if self.save_path:
         trial = "" if self.trial is None else f".t{self.trial}"
         fname = (
            f"results"
            f"_{self.height}x{self.width}x{self.depth}"
            f"@{self.connections.value}"
            f"-{self.branches.__class__.__name__}"
            f".b{self.scheme.batch_size}"
            f"{trial}"
            f".pkl"
         )
         fname = Path(self.save_path) / fname
         self.results.save(fname)

      #return valid_loss

   def run_epochs(self, n: int, webhook: Optional[str]=None) -> None:
      """Train for ``n`` epochs.

      Args:
         n: The number of epochs to train for.
         webhook: The Discord or Slack webhook URL to post to. See `here`_ for what it looks like.
         
      .. _here: https://i.imgur.com/Z8qiTE2.png
      """
      if n <= 0: return

      for e in range(n):
         self.run_epoch()

      # webhook code
      if webhook:
         total_time = round(sum(self.results.train_times) + sum(self.results.valid_times))
         epochs = f"{n} epoch" + ("s" if n != 1 else "")
         content = (
            f"Finished training `{repr(self)}` for {epochs}! "
               f"(took around {total_time} seconds total)\n"
            f"The epoch with best performance was epoch {self.results.best_epoch}:\n"
            f"* Validation loss: {round(self.results.best_valid_loss,2)}\n"
            f"* F1 score: {round(self.results.f1_scores[self.results.best_epoch],3)}\n"
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
   pass