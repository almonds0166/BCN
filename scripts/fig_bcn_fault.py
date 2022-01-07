
"""
Make a figure displaying different levels of fault.

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

from bcn import Fault

from plotutils import Colors, Cmaps, ABBREVIATIONS, save_fig

FONT_SIZE = 22
plt.rcParams.update({'font.size': FONT_SIZE})

PERCENTAGES = (2, 10,)
WIDTHS = (16, 30,)

PADDING = 1

def main():
   fig, axes = plt.subplots(len(WIDTHS), len(PERCENTAGES), figsize=(10, 10))
   axes = axes.flatten()

   k = 0
   for i, width in enumerate(WIDTHS):
      for j, percentage in enumerate(PERCENTAGES):
         ax = axes[k]; k += 1

         p = PADDING
         m = np.ones((width, width))
         inner = np.zeros((width-2*p, width-2*p))
         m[p:-p, p:-p] = inner

         fault = Fault(
            proportion=percentage/100,
            width=width,
            depth=2,
            padding=p
         )

         f = ~fault[0].reshape((width, width))

         ax.imshow(m, cmap=Cmaps.GRAY, vmax=2)
         ax.imshow(f, cmap=Cmaps.RED)
         ax.set_xticks(())
         ax.set_yticks(())
         ax.set_xticks(np.arange(-.5, width, 1), minor=True)
         ax.set_yticks(np.arange(-.5, width, 1), minor=True)
         ax.grid(which="minor", color="k", linestyle=":")

         #ax.set_title(f"{width}x{width}, {percentage}% fault")
         if i == len(WIDTHS)-1: ax.set_xlabel(f"{percentage}% fault")
         if j == 0: ax.set_ylabel(f"{width}x{width}")


   #plt.tight_layout()
   save_fig(plt, "./fig_bcn_fault/", "fault-masks.png", show=True)

if __name__ == "__main__":
   main()