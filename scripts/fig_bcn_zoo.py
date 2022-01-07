
"""
Make a figure displaying the eight sets of branches I worked with.

Figure outputs are saved under ./fig_bcn_zoo/

This is for chapter 3 of my thesis.

Improvements:
- Add a fudge factor to get away from 0
- Shave off outer ring, so it's 7x7
- Increase darkness of grid
- Blue for center connection
"""

import sys; sys.path.append("../")
from pathlib import Path
import re

import torch
import numpy as np
import matplotlib.pyplot as plt

from bcn.branches import DirectOnly
from bcn.branches.uniform import NearestNeighbor, NearestNeighborOnly, NextToNN, NextToNNOnly
from bcn.branches.informed import Kappa, IndirectOnly

from plotutils import Colors, Cmaps, ABBREVIATIONS, BRANCH_NAMES

FONT_SIZE = 22
plt.rcParams.update({'font.size': FONT_SIZE})

BRANCHES = [
   DirectOnly(), Kappa(1.0), Kappa(1.5), IndirectOnly(),
   NearestNeighborOnly(), NextToNNOnly(), NearestNeighbor(), NextToNN(),
]

WIDTH = 7
TEXT = False

def truncate_edges(branches, width):
   """Truncate the edges of a given matrix to the given width
   """
   assert width % 2, "given width must be odd"
   w = len(branches)
   r = (w - width) // 2
   return branches[r:-r,r:-r]

def main():
   vmin = float("inf")
   vmax = -float("inf")

   fig, axes = plt.subplots(2,4, figsize=(18,9))

   for ax, branches in zip(axes.flatten(), BRANCHES):
      m = truncate_edges(branches[0,0], WIDTH)

      # color direct and indirect differently
      o = WIDTH//2
      direct = torch.zeros(m.shape)
      direct[o,o] = m[o,o]
      indirect = torch.zeros(m.shape)
      for r in range(WIDTH):
         for c in range(WIDTH):
            if (r,c) == (o,o): continue
            indirect[r,c] = m[r,c]

      # plot
      vmin = torch.min(m)
      vmax = torch.max(m)
      ax.imshow(indirect, cmap=Cmaps.RED, vmin=vmin, vmax=vmax)
      ax.imshow(direct, cmap=Cmaps.RED, vmin=vmin, vmax=vmax)
      #max_ = max(abs(vmin), vmax)
      #ax.imshow(m, cmap=Cmaps.BLUE_TO_RED, vmin=-max_, vmax=max_)

      # icing
      width = WIDTH #branches.width
      ax.set_xticks(())
      ax.set_yticks(())
      ax.set_xticks(np.arange(-.5, width, 1), minor=True)
      ax.set_yticks(np.arange(-.5, width, 1), minor=True)
      ax.grid(which="minor", color="k", linestyle=":")

      if TEXT:
         for i in range(width):
            for j in range(width):
               val = m[i,j].item()
               if abs(val) >= 1e-2 and i >= width//2 and i == j:
                  ax.text(i, j, f"{val:.2f}",
                     ha="center", va="center", color="k",
                     fontsize="small",
                  )

      b = str(branches)
      title = f"{ABBREVIATIONS[b]}"
      ax.set_title(title)

   fig.suptitle("Sets of branches investigated")

   plt.tight_layout()

   output_folder = Path("./fig_bcn_zoo/")
   output_folder.mkdir(parents=True, exist_ok=True)
   filename = f"bcn-zoo.png"
   plt.savefig(output_folder / filename)
   plt.show()

if __name__ == "__main__":
   _ = main()