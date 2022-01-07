"""
Make a figure of the "keypad connectedness" of each layer in a specified model.

I'm not sure of a better term, but for now it's called "keypad connectedness"
and it represents the number of connections each neuron in a layer is capable
of making to the output keypad. This number ranges from 0 to 10 inclusive.

Figure outputs are saved under ./fig_bcn_cores/

This is for chapter 3 of my thesis.

Ideas for improvement:
- add colorbar
- remove outer whitespace
"""

import sys; sys.path.append("../")
from pathlib import Path
import re

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from bcn import BCN, Connections
from bcn.branches import DirectOnly
from bcn.branches.uniform import ( NearestNeighborOnly, NextToNNOnly,
                                   NearestNeighbor, NextToNN, )
from bcn.branches.informed import Kappa, IndirectOnly
from plotutils import keypad_connectedness, ABBREVIATIONS, BRANCH_NAMES

CONNECTIONS = Connections.ONE_TO_9
WIDTH = 16
BRANCHES = (DirectOnly(), NearestNeighbor())#, NextToNN())
DEPTH = 6

MARKS = {(0,0),(1,3),}
#MARKS = {(1,0),(2,2),}

FONT_SIZE = 24
plt.rcParams.update({'font.size': FONT_SIZE})

def main():

   b = len(BRANCHES)
   br_names = [ABBREVIATIONS[repr(br).strip("()")] for br in BRANCHES]
   fig, axes = plt.subplots(b, DEPTH, figsize=(3.6*DEPTH,3.7*b))

   for i, branches in enumerate(BRANCHES):
      model = BCN(WIDTH, DEPTH,
         connections=CONNECTIONS,
         branches=branches,
         dropout=0,
      )
      images = keypad_connectedness(model)
      for j, image in enumerate(images):
         ax = axes[i,j]
         colors = [(0,.53,.74,c) for c in np.linspace(0,1,100)]
         cmapblue = mcolors.LinearSegmentedColormap.from_list("mycmap", colors, N=10)
         last_im = ax.imshow(image, cmap=cmapblue, vmin=0, vmax=10)
         ax.set_xticks(tuple())
         ax.set_yticks(tuple())
         ax.set_xticks(np.arange(-.5, WIDTH, 1), minor=True)
         ax.set_yticks(np.arange(-.5, WIDTH, 1), minor=True)
         ax.grid(which="minor", color="darkgray", linestyle=":")
         if j == 0:
            ax.set_ylabel(f"{br_names[i]}")

         if (i,j) in MARKS:
            ax.set_xlabel("*", fontdict={'fontsize': 38})
         elif j == DEPTH-1:
            ax.set_xlabel("Penultimate plane", fontdict={'fontsize': 19})


   # add colorbar
   fig.subplots_adjust(right=0.85)
   cbar_ax = fig.add_axes([0.89, 0.2, 0.015, 0.6])
   cbar = fig.colorbar(last_im, cax=cbar_ax, ticks=np.arange(0, 11))
   cbar.ax.tick_params(labelsize=17)

   if len(br_names) == 1:
      branch_names = br_names[0]
   elif len(br_names) == 2:
      branch_names = f"{br_names[0]} & {br_names[1]}"
   else:
      branch_names = ", ".join(br_names[:-1]) + f", & {br_names[-1]}"
      
   fig.suptitle((
      f"Output connectedness of {branch_names} "
      f"on {WIDTH}x{WIDTH} models"
   ))
   #fig.tight_layout()

   output_folder = Path("./fig_bcn_cores/")
   output_folder.mkdir(parents=True, exist_ok=True)
   filename = f"bcn-cores_{WIDTH}x{WIDTH}x{DEPTH}@{CONNECTIONS.value}.png"
   plt.savefig(output_folder / filename)
   plt.show()

   caption = (
      f"Output connectedness images of {branch_names} models "
      f"for input dimensions {WIDTH}x{WIDTH}."
   )

   lines = [
      "% Generated with ``scripts/fig_bcn_cores.py``",
      "\\begin{figure}[h]",
      "   \\centering",
      f"   \\includegraphics[width=\\textwidth]{{{filename}}}",
      f"   \\caption[{caption}]{{{caption}}}",
      f"   \\label{{fig:bcn_cores_{WIDTH}x{WIDTH}x{DEPTH}@{CONNECTIONS.value}}}",
      "\\end{figure}",
   ]

   code = "\n".join(lines)
   print("```latex")
   print(code)
   print("```")

   return code


if __name__ == '__main__':
   _ = main()
