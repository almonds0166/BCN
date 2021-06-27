
from typing import Optional, Dict, Tuple

import torch
import numpy as np

class Branches:
   """Base class representing branching connections.

   Each branching (indirect) connection is encoded by a matrix where each element represents the
   connection from a previous layer to the following layer. The center of this matrix represents
   the connection from a previous layer's neuron to the single nearest neighbor neuron in the next
   layer. Likewise, off center elements represent connections from a previous layer's neuron to off
   center nearest neighbor neurons. Class instances act the same as Python dicts.

   Note:
      Directly index into this class's instances instead of using the `Branches.connections` dict.
   
   Args:
      width: Size of the matrices representing the branches. Default 9.

   Attributes:
      width (int): Size of the matrices representing the branches.
      center (int): index of the center of the matrices representing the branches.
      connections (Dict[Tuple[int,int],torch.Tensor]): Map from 2-dimensional offset to the matrix
         that represents the indirect connections/convolution kernel.
      default (torch.Tensor): The matrix to use when an offset isn't defined in
         `Branches.connections`.
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

   def normalize(self, norm: float=1) -> None:
      """Normalize the sum of the connections to a norm.

      Args:
         norm: The norm to use, default 1.
      """
      assert norm > 0, f"norm must be a positive float; given: `{norm}`."

      sum_ = torch.sum(self.default)
      if sum_ != 0:
         self.default = norm * self.default / sum_

      for k, v in self.connections.items():
         sum_ = torch.sum(v)
         if sum_ != 0:
            self.connections[k] = norm * v / sum_

   @staticmethod
   def pan(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
      """Pan a tensor ``x`` down ``dy`` and over ``dx``.

      Similar to torch.roll, circularly convolves the given tensor, instead with zero padding.

      Args:
         x: 2D matrix to pan.
         dy: Count of places to shift the tensor downward.
         dx: Count of places to shift the tensor rightward.

      Returns:
         torch.Tensor: Panned/shifted matrix.
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

   For direct-only connections, each arm has only one finger. This is more or less equivalent to a
   standard dense neural network layer.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # there's nothing necessary to change here since the default is a direct-only kernel

   def __repr__(self):
      return f"{self.__class__.__name__}()"

class Vital(Branches):
   """Connection matrices designed by Inoela Vital and Cardinal Warde ca. 2021.

   Based off of the empirical numbers obtained by William "Bill" Herrington ca. 2015.

   Note:
      This is used with 1-to-9 connections or 1-to-25 connections only.
   """
   def __init__(self):
      super().__init__(width=9)
      self.default = torch.zeros((9,9))
      # center
      self.connections[0,0] = torch.tensor([
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,  0.1,  0.3,  0.1, 0, 0, 0 ],
         [ 0, 0, 0, 0.35, 38.6, 0.35, 0, 0, 0 ],
         [ 0, 0, 0,  0.1,  0.3,  0.1, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
      ])
      # edge (1-to-9)
      self.connections[0,1] = torch.tensor([
         [ 0, 0,   0,   0,   0,    0,   0, 0, 0 ],
         [ 0, 0,   0,   0,   0,    0,   0, 0, 0 ],
         [ 0, 0,   0,   0,   0,    0,   0, 0, 0 ],
         [ 0, 0,   0, 0.2, 0.2,    0,   0, 0, 0 ],
         [ 0, 0, 0.1, 1.3, 5.7, 23.7, 0.8, 0, 0 ],
         [ 0, 0,   0, 0.2, 0.2,    0,   0, 0, 0 ],
         [ 0, 0,   0,   0,   0,    0,   0, 0, 0 ],
         [ 0, 0,   0,   0,   0,    0,   0, 0, 0 ],
         [ 0, 0,   0,   0,   0,    0,   0, 0, 0 ],
      ])
      self.connections[-1, 0] = torch.rot90(self.connections[0,1], 1, (0, 1))
      self.connections[ 0,-1] = torch.rot90(self.connections[0,1], 2, (0, 1))
      self.connections[ 1, 0] = torch.rot90(self.connections[0,1], 3, (0, 1))
      # diagonal (1-to-9)
      self.connections[-1, 1] = torch.tensor([
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0, 0.65, 21.4, 0, 0, 0 ],
         [ 0, 0, 0, 0.35,  1.7, 0.65, 0, 0, 0 ],
         [ 0, 0, 0,  0.3, 0.35,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
         [ 0, 0, 0,    0,    0,    0, 0, 0, 0 ],
      ])
      self.connections[-1,-1] = torch.rot90(self.connections[-1,1], 1, (0, 1))
      self.connections[ 1,-1] = torch.rot90(self.connections[-1,1], 2, (0, 1))
      self.connections[ 1, 1] = torch.rot90(self.connections[-1,1], 3, (0, 1))
      # far edge (1-to-25)
      self.connections[0,2] = torch.tensor([
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
         [ 0, 0, 0,   0, 0.2, 0.2,    0,   0, 0 ],
         [ 0, 0, 0, 0.1, 1.3, 5.7, 23.7, 0.8, 0 ],
         [ 0, 0, 0,   0, 0.2, 0.2,    0,   0, 0 ],
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
      ])
      self.connections[-2, 0] = torch.rot90(self.connections[0,2], 1, (0, 1))
      self.connections[ 0,-2] = torch.rot90(self.connections[0,2], 2, (0, 1))
      self.connections[ 2, 0] = torch.rot90(self.connections[0,2], 3, (0, 1))
      # edge-diagonal (1-to-25)
      self.connections[-1,2] = torch.tensor([
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
         [ 0, 0, 0,   0,   0,   0,    0, 0.8, 0 ],
         [ 0, 0, 0,   0,   0, 0.2, 23.7,   0, 0 ],
         [ 0, 0, 0,   0, 0.2, 5.7,  0.2,   0, 0 ],
         [ 0, 0, 0,   0, 1.3, 0.2,    0,   0, 0 ],
         [ 0, 0, 0, 0.1,   0,   0,    0,   0, 0 ],
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
         [ 0, 0, 0,   0,   0,   0,    0,   0, 0 ],
      ])
      self.connections[-2,-1] = torch.rot90(self.connections[-1,2], 1, (0, 1))
      self.connections[ 1,-2] = torch.rot90(self.connections[-1,2], 2, (0, 1))
      self.connections[ 2, 1] = torch.rot90(self.connections[-1,2], 3, (0, 1))
      self.connections[-2, 1] = torch.transpose(self.connections[-1,2], 0, 1)
      self.connections[-1,-2] = torch.rot90(self.connections[-2,1], 1, (0, 1))
      self.connections[ 2,-1] = torch.rot90(self.connections[-2,1], 2, (0, 1))
      self.connections[ 1, 2] = torch.rot90(self.connections[-2,1], 3, (0, 1))
      # diagonal (1-to-25)
      self.connections[-2,2] = torch.tensor([
         [ 0, 0, 0, 0,   0,   0,    0, 0, 0 ],
         [ 0, 0, 0, 0,   0,   0,    0, 0, 0 ],
         [ 0, 0, 0, 0,   0, 0.3, 21.4, 0, 0 ],
         [ 0, 0, 0, 0, 0.2, 1.7,    1, 0, 0 ],
         [ 0, 0, 0, 0, 0.3, 0.5,    0, 0, 0 ],
         [ 0, 0, 0, 0,   0,   0,    0, 0, 0 ],
         [ 0, 0, 0, 0,   0,   0,    0, 0, 0 ],
         [ 0, 0, 0, 0,   0,   0,    0, 0, 0 ],
         [ 0, 0, 0, 0,   0,   0,    0, 0, 0 ],
      ])
      self.connections[-2,-2] = torch.rot90(self.connections[-2,2], 1, (0, 1))
      self.connections[ 2,-2] = torch.rot90(self.connections[-2,2], 2, (0, 1))
      self.connections[ 2, 2] = torch.rot90(self.connections[-2,2], 3, (0, 1))

      self.normalize()