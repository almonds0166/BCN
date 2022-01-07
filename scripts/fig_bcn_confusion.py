
"""
Make a figure of confusion matrices given the results. Actual class is on the
left side (rows), and Predicted class is on the top (columns). Uses the
average of all found trials.

Figure outputs are saved under ./fig_bcn_confusion/

This is for chapter 3 of my thesis.

Ideas for improvement:
- Better color scheme
"""

import sys; sys.path.append("../")
from pathlib import Path
import re

import torch
import numpy as np
import matplotlib.pyplot as plt

from bcn import Results, Dataset, Connections
from bcn import BCN, TrainingScheme
from bcn.branches import DirectOnly
from bcn.branches.uniform import NearestNeighbor, NearestNeighborOnly, NextToNN, NextToNNOnly
from bcn.branches.informed import Kappa, IndirectOnly

from plotutils import ABBREVIATIONS

RESULTS_PATH = Path(input("Enter the location of all your results\n> "))

# choose model
DATASET = Dataset.FASHION
CONNECTIONS = Connections.ONE_TO_9
BATCH_SIZE = 64
W = 30
H = W
D = 3

TRIAL = None # ``None`` for all trials

FONT_SIZE = 24
plt.rcParams.update({'font.size': FONT_SIZE})

BRANCHES = {
   "DirectOnly": DirectOnly(),
   "informed.Kappa(1.0)": Kappa(1.0),
   "informed.Kappa(1.5)": Kappa(1.5),
   "informed.IndirectOnly": IndirectOnly(),
   "uniform.NearestNeighborOnly": NearestNeighborOnly(),
   "uniform.NextToNNOnly": NextToNNOnly(),
   "uniform.NearestNeighbor": NearestNeighbor(),
   "uniform.NextToNN": NextToNN(),
}

FASHION_CLASSES = [
   "T-shirt/top",
   "Trouser",
   "Pullover",
   "Dress",
   "Coat",
   "Sandal",
   "Shirt",
   "Sneaker",
   "Bag",
   "Ankle boot",
]

def mean(ell, default=None):
   return sum(ell) / len(ell) if ell else default

def main():

   data = {}

   for file in RESULTS_PATH.iterdir():
      if not file.name.startswith("weights_") or file.suffix != ".pt": continue
      if not f".{DATASET.name}." in file.stem: continue
      if not f"@{CONNECTIONS.value}-" in file.stem: continue
      if not f".b{BATCH_SIZE}." in file.stem: continue
      if TRIAL is not None and not file.stem.endswith(f".t{TRIAL}"): continue

      m = re.search(r"([0-9]+)x([0-9]+)x([0-9]+)", file.stem)
      h = int(m.group(1))
      w = int(m.group(2))
      d = int(m.group(3))

      if f"{h}x{w}x{d}" != f"{H}x{W}x{D}": continue

      print(file.name)

      m = re.search(rf"@{CONNECTIONS.value}-([\w\.\(\)0-9]+).{DATASET.name}.", file.stem)
      b = m.group(1)

      bucket = (h, w, d, b)

      if not bucket in data:
         data[bucket] = []

      model = BCN(w, d,
         branches=BRANCHES[b],
         connections=CONNECTIONS,
      )
      scheme = TrainingScheme(
         dataset=DATASET,
         batch_size=BATCH_SIZE,
         width=w,
         padding=1,
      )
      model.train(
         scheme=scheme,
         from_weights=file,
      )

      data[bucket].append(
         model.confusion()
      )

   vmax = 0
   for bucket in data.values():
      for confusion in bucket:
         vmax = max(vmax, torch.max(confusion))

   #print("vmax is", vmax)

   trials = float("inf") # minimum number of trials encountered

   # set up figure
   fig, axes = plt.subplots(2,4)
   fig.set_size_inches(20,10)
   axes = axes.flatten()

   for i, b in enumerate(BRANCHES.keys()):
      bucket = (H, W, D, b)
      trials = min(trials, len(data[bucket]))
      confusion = mean(data[bucket])

      ax = axes[i]
      last_im = ax.imshow(confusion, vmin=0, vmax=vmax)
      ax.set_title(ABBREVIATIONS[b])
      ax.set_yticks(())
      ax.set_xticks(())

      if i in (0, 4):
         ax.set_ylabel("Actual class")
         ax.set_yticks(range(10))
         ax.set_yticklabels(FASHION_CLASSES if DATASET == Dataset.FASHION else range(10))

      if i >= 4:
         ax.set_xlabel("Predicted class")
         ax.set_xticks(range(10))
         if DATASET == Dataset.FASHION:
            ax.set_xticklabels(FASHION_CLASSES, rotation=80, fontdict={'fontsize': 20})
         else:
            ax.set_xticklabels(range(10))

   # more room for Fashion
   if DATASET == Dataset.FASHION:
      fig.subplots_adjust(bottom=0.25)

   # add colorbar
   fig.subplots_adjust(right=0.85)
   cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.6])
   lower = 0
   upper = 100 * round(vmax.item()/100)
   cbar = fig.colorbar(last_im, cax=cbar_ax, ticks=np.arange(lower, upper, 100))

   # tidy up whole plot
   fig.suptitle((
      f"{H}x{W}x{D}, 1-to-{CONNECTIONS.value} interclass confusion matrices "
      f"on {DATASET.value}"
   ))
   output_folder = Path("./fig_bcn_confusion/")
   output_folder.mkdir(parents=True, exist_ok=True)
   filename = f"bcn-confusion_{H}x{W}x{D}_{DATASET.name.lower()}@{CONNECTIONS.value}.png"
   #plt.tight_layout()
   plt.savefig(output_folder / filename)
   #plt.show()

   caption = (
      f"Interclass confusion matrices for {H}x{W}x{D}-sized models on {DATASET.value} "
      f"at 1-to-{CONNECTIONS.value} connections across the eight types of branches investigated "
      f"({trials} trials each). Columns represent the classes as predicted by the model; "
      f"rows represent the correct class."
   )

   lines = [
      "% Generated with ``scripts/fig_bcn_confusion.py``",
      "\\begin{figure}[h]",
      "   \\centering",
      f"   \\includegraphics[width=\\textwidth]{{{filename}}}",
      f"   \\caption{{{caption}}}",
      f"   \\label{{fig:confusion_{H}x{W}x{D}_{DATASET.name.lower()}@{CONNECTIONS.value}}}",
      "\\end{figure}",
   ]

   code = "\n".join(lines)
   print("```latex")
   print(code)
   print("```")

   return code

if __name__ == '__main__':
   _ = main()

