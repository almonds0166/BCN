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
from plotutils import keypad_connectedness

CONNECTIONS = Connections.ONE_TO_9
BRANCHES = NearestNeighborOnly()
DEPTH = 6

BRANCH_NAMES = {
   "DirectOnly": "Direct connections only",
   "informed.Kappa(1.0)": "Grating strength 1.0",
   "informed.Kappa(1.5)": "Grating strength 1.5",
   "informed.IndirectOnly": "Grating strength 2.4048",
   "uniform.NearestNeighborOnly": "First ring only",
   "uniform.NextToNNOnly": "Second ring only",
   "uniform.NearestNeighbor": "Uniform nearest neighbor",
   "uniform.NextToNN": "Uniform next-to-nearest neighbor",
}

FONT_SIZE = 14
plt.rcParams.update({'font.size': FONT_SIZE})

def main():

   fig, axes = plt.subplots(2, DEPTH, figsize=(4*DEPTH,8))
   
   for i, width in enumerate((16, 30)):
      model = BCN(width, DEPTH,
         connections=CONNECTIONS,
         branches=BRANCHES,
         dropout=0,
      )
      images = keypad_connectedness(model)
      for j, image in enumerate(images):
         ax = axes[i,j]
         colors = [(0,.53,.74,c) for c in np.linspace(0,1,100)]
         cmapblue = mcolors.LinearSegmentedColormap.from_list("mycmap", colors, N=5)
         ax.imshow(image, cmap=cmapblue, vmin=0, vmax=10)
         ax.set_xticks(tuple())
         ax.set_yticks(tuple())
         ax.set_xticks(np.arange(-.5, width, 1), minor=True)
         ax.set_yticks(np.arange(-.5, width, 1), minor=True)
         ax.grid(which="minor", color="lightgray", linestyle=":")

   name = BRANCH_NAMES[str(BRANCHES)]
   fig.suptitle((
      f"Keypad connectedness of {name.lower()} branches\n"
      f"of shapes 16x16x{DEPTH} and 30x30x{DEPTH}"
   ))

   output_folder = Path("./fig_bcn_cores/")
   output_folder.mkdir(parents=True, exist_ok=True)
   filename = f"bcn-cores_{BRANCHES}_x{DEPTH}@{CONNECTIONS.value}.png"
   plt.savefig(output_folder / filename)
   plt.show()

   caption = (
      f"``Keypad connectedness'' images of {name} models "
      f"for shapes 16x16x{DEPTH} and 30x30x{DEPTH}."
   )

   lines = [
      "% Generated with ``scripts/fig_bcn_cores.py``",
      "\\begin{figure}[h]",
      "   \\centering",
      f"   \\includegraphics[width=\\textwidth]{{{filename}}}",
      f"   \\caption{{{caption}}}",
      f"   \\label{{fig:bcn_cores_{BRANCHES}_x{DEPTH}@{CONNECTIONS.value}}}",
      "\\end{figure}",
   ]

   code = "\n".join(lines)
   print("```latex")
   print(code)
   print("```")

   return code


if __name__ == '__main__':
   _ = main()
