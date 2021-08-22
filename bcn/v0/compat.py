
from pathlib import Path
from typing import Union, Optional
from collections import OrderedDict
import re
import math

import torch

from .. import Results as Results_

# old Results class
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
      best_epoch (int): Index corresponding to the minimum of `Results.valid_losses`.
      tag (str): Anything notable about the model or results. Used as plot titles when plotting.
         Set via the `BCN.train` method.
      version (str): The version of the BCN Python module that these results came from.
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
      self.tag = ""
      self.version = "0.4.98"

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

def migrate_results(
   file: Union[Path,str],
   new_filename: Optional[Union[Path,str]]=None) \
-> None:
   """Migrate a results file that was generated from v0.4.98 or earlier to a v1 results file.

   This overwrites the file unless the ``new_filename`` parameter is used.

   Example usage: ::

      >>> from pathlib import Path
      >>> from bcn.v0 import migrate_results
      >>> for file in Path("./results/").iterdir():
      ...    migrate_results(file)
      ... 

   Args:
      file: The v0 results file to convert.
      new_filename: The new filename to use. Default is to use the original filename.
   """
   file = Path(file)

   r = Results()
   r.load(file)

   r_ = Results_()
   r_.__dict__.update(r.__dict__)

   # replace `best_epoch` with `best`
   r_.best = max(range(len(r_.f1_scores)), key=lambda i: r_.f1_scores[i])
   del r_.__dict__["best_epoch"]
   del r_.__dict__["best_valid_loss"]

   # version info is now kept in a set
   r_.versions = set((r.version,))
   del r_.__dict__["version"]

   # devices are unknown :/
   r_.devices = set()

   # no weight perturbation yet
   r_.step = 0
   r_.wp_layers = []

   if new_filename is None: new_filename = Path(filename)

   r_.save(new_filename)

def migrate_weights(
   file: Union[Path,str],
   new_filename: Optional[Union[Path,str]]=None) \
-> None:
   """Migrate a weights file that was generated from v0.4.98 or earlier to a v1 weights file.

   This overwrites the file unless the ``new_filename`` parameter is used.

   Example usage: ::

      >>> from pathlib import Path
      >>> from bcn.v0 import migrate_weights
      >>> for file in Path("./results/").iterdir():
      ...    migrate_weights(file)
      ... 

   Args:
      file: The v0 weights file to convert.
      new_filename: The new filename to use. Default is to use the original filename.
   """
   file = Path(file)

   W = torch.load(file, map_location="cpu")

   W_ = OrderedDict()

   # get file info using regex
   # example: weights_30x30x6@9-informed.IndirectOnly.MNIST.b64.t1.pt
   m = re.match(
      r"weights_([0-9]+)x([0-9]+)x([0-9]+)@([0-9]+)-" \
      r"(\w+\.?\w+)\.(\w+)\.b([0-9]+)(\.t(?:\w+))?\.pt",
      file.name
   )

   # detect information from filename
   left, _ = file.name.split("-")
   shape = left[left.rfind("_")+1:]
   size, c = shape.split("@")
   h, w, d = size.split("x")

   h = int(h)
   w = int(w)
   hw = h*w

   d = int(d)
   c = int(c)

   ell = (int(math.sqrt(c))-1)//2
   ells = range(-ell, ell+1)

   for l in range(d):
      W_[f"layers.{l}.weights"] = torch.zeros((c,hw,1))

      index = 0
      for dy in ells:
         for dx in ells: # row-major
            W_[f"layers.{l}.weights"][index,:,:] = W[f"layers.{l}.({dy},{dx})"]
            index += 1

   torch.save(W_, new_filename)


