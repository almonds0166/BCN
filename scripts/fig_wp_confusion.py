
"""
Interclass confusion figures for chapter 4. Before and after weight perturbation recovery.
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

from plotutils import ABBREVIATIONS, ABBREVIATIONS_SAFE, save_fig

BATCH_SIZE = 64
PERCENT = 10

RESULTS_PATH = Path(input("Enter the location of all your *baseline* results\n> "))
WP_PATH = Path(input("Enter the location of all your *WP* results\n> "))
WP_PATH /= f"{PERCENT}percent" # get subfolder with only the desired results

# choose model
DATASET = Dataset.MNIST
CONNECTIONS = Connections.ONE_TO_9
BRANCHES = "uniform.NearestNeighborOnly"
HEIGHT = 30
DEPTH = 3
TRIAL = 2

FONT_SIZE = 20
plt.rcParams.update({'font.size': FONT_SIZE})

BRANCHES_ = {
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

def load_from_path(path):
   model = BCN(HEIGHT, DEPTH,
      branches=BRANCHES_[BRANCHES],
      connections=CONNECTIONS,
   )
   scheme = TrainingScheme(
      dataset=DATASET,
      batch_size=BATCH_SIZE,
      width=HEIGHT,
      padding=1,
   )
   model.train(
      scheme=scheme,
      from_path=path, # !!
      trial=TRIAL,
   )
   return model

def plot_recovered_confusion():
   # set up figure
   fig, axes = plt.subplots(1,2)
   fig.set_size_inches(16,11)

   left, right = axes.flatten()

   fname = (
      f"fault_{HEIGHT}x{HEIGHT}x{DEPTH}"
      f"@{CONNECTIONS.value}-{BRANCHES}"
      f".{DATASET.name}.b{BATCH_SIZE}.t{TRIAL}.pkl"
   )
   fault = torch.load(WP_PATH / fname)

   # original model with fault
   model = load_from_path(RESULTS_PATH)
   confusion_l = model.confusion(fault=fault)
   # after recovery
   model = load_from_path(WP_PATH)
   confusion_r = model.confusion(fault=fault)

   vmax = max(torch.max(confusion_l), torch.max(confusion_r))

   im_l = left.imshow(confusion_l, vmin=0, vmax=vmax)
   left.set_title("Before recovery")

   im_r = right.imshow(confusion_r, vmin=0, vmax=vmax)
   right.set_title("After recovery")

   # tidy up plots
   for i, ax in enumerate(axes.flatten()):
      ax.set_yticks(())
      ax.set_xticks(())

      if i == 0 or DATASET == Dataset.MNIST:
         ax.set_ylabel("Actual class")
         ax.set_yticks(range(10))
         ax.set_yticklabels(FASHION_CLASSES if DATASET == Dataset.FASHION else range(10))

      ax.set_xlabel("Predicted class")
      ax.set_xticks(range(10))
      if DATASET == Dataset.FASHION:
         ax.set_xticklabels(FASHION_CLASSES, rotation=70)
      else:
         ax.set_xticklabels(range(10))

   # add colorbar
   fig.subplots_adjust(right=0.85)
   cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.6])
   lower = 0
   upper = 100 * round(vmax.item()/100)
   cbar = fig.colorbar(im_r, cax=cbar_ax, ticks=np.arange(lower, upper, 100))


   title = (
      #f"Interclass confusion matrices before and after weight perturbation recovery\n"
      f"{HEIGHT}x{HEIGHT}x{DEPTH} 1-to-{CONNECTIONS.value} "
      f"with {ABBREVIATIONS[BRANCHES]} branches evaluated on {DATASET.value} (trial {TRIAL})"
   )
   plt.suptitle(title, y=0.9)

   return plt

def latex(filename):
   short_caption = f"Interclass confusion matrices before and after weight perturbation"
   caption = f"Interclass confusion matrices before and after weight perturbation recovery."

   lines = [
      "\\begin{figure}[H]",
      "\\centering",
      f"\\includegraphics[width=\\textwidth]{{{filename}}}",
      f"\\caption[{short_caption}]{{{caption}}}",
      f"\\label{{fig:{filename[:-4]}}}",
      "\\end{figure}",
   ]

   return "\n".join(lines)

def main():

   plt = plot_recovered_confusion()
   #plt.tight_layout()
   
   fname = (
      f"{HEIGHT}x{HEIGHT}x{DEPTH}@{CONNECTIONS.value}"
      f"-{ABBREVIATIONS_SAFE[BRANCHES]}.{DATASET.name}.t{TRIAL}.confusion.png"
   )

   save_fig(plt, f"fig_wp_confusion/{PERCENT}percent", fname, True)

   code = latex(fname)

   print("```latex")
   print(code)
   print("```")

   return code

if __name__ == '__main__':
   _ = main()

