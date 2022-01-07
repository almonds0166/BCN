
"""
Generate a figure of four subplot bar plots of weight perturbation improvements. One for depth 3
MNIST, one for depth 6 MNIST, one for depth 3 Fashion-MNIST, one for depth 6 Fashion-MNIST.

This is for chapter 4 of my thesis.

The given path for the *WP* results should contain the following structure:

   ./{p}percent/...

where the inner-most folder contains all the pkl & pt files, and ``p`` represents the percentage of
fault, e.g. 2 or 10 for 2% or 10% respectively. Remember that the file that had WP ran on it have
trial names like "t1" or "t3", but the ones that had SGD ran on them have trial names like "t1o"
and "t2o", where the "o" kind of represents an origin or baseline or control group.

Ideas I have in mind for improvement:
- Combine the two datasets into another set of bar plots.
"""

import sys; sys.path.append("../")
from pathlib import Path
import re

from matplotlib import pyplot as plt
import numpy as np

from bcn import Results, Dataset, Connections

from plotutils import MNIST_COLOR, FASHION_COLOR

WP_PATH = Path(input("Enter the location of your *WP* results\n> "))

CONNECTIONS = Connections.ONE_TO_9
PERCENT = 2

# get subfolder with only the desired results
WP_PATH /= f"{PERCENT}percent"

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

FONT_SIZE = 20
plt.rcParams.update({'font.size': FONT_SIZE})

DX = 0.15

def mean(ell, default=None):
   return sum(ell) / len(ell) if ell else default

def count_successul_perturbations(dataset, depth, debug=False):
   """Return a list of all tallied WP improvement with respect to each layer.

   Uses the WP_PATH as defined in the global frame above.
   """
   improvements = [0] * depth
   margins = [0] * depth

   trials = 0

   for file in WP_PATH.iterdir():
      if not file.name.startswith("results_") or file.suffix != ".pkl": continue
      if not f".{dataset.name}." in file.stem: continue
      if not f"@{CONNECTIONS.value}-" in file.stem: continue
      if file.stem.endswith("o"): continue # just skip the SGD ones

      m = re.search(r"([0-9]+)x([0-9]+)x([0-9]+)", file.stem)
      h = int(m.group(1))
      w = int(m.group(2))
      d = int(m.group(3))

      if d != depth: continue

      trials += 1

      if debug: print(file.name)

      m = re.search(rf"@{CONNECTIONS.value}-([\w\.\(\)0-9]+).{dataset.name}.", file.stem)
      b = m.group(1)

      r = Results()
      r.load(file)

      N = len(r.wp_layers)
      f1_scores = r.f1_scores[-(N+1):]

      for i, (l, f1) in enumerate(zip(r.wp_layers, f1_scores)):
         next_f1 = f1_scores[i+1]
         margin = next_f1 - f1
         margins[l] += margin # accumulate absolute change in F1 score!
         improvements[l] += (margin > 0) # record all *improvements*!

   return improvements, margins, trials

def plot_bins(ax, bins1, bins2):
   assert len(bins1) == len(bins2)
   N = len(bins1)
   x = np.arange(N)
   ax.bar(x-DX, bins1, width=2*DX, color=MNIST_COLOR, align="center", label="MNIST")
   ax.bar(x+DX, bins2, width=2*DX, color=FASHION_COLOR, align="center", label="Fashion-MNIST")

def plot_subfigures(index):
   YLABELS = ("Total count of improvements", "Total $F_1$ score margin")
   TITLES = ("Weight perturbation successes counted across respective layers",
      "Change in $F_1$ score accumulated across respective layers")

   fig, axes = plt.subplots(1,2,gridspec_kw={"width_ratios": [1,1.618]})
   fig.set_size_inches(16,6)

   max_y_imp = 0
   max_y_mar = 0
   data = {}
   trials = 0

   for dataset in (Dataset.MNIST, Dataset.FASHION):
      for depth in (3, 6):
         improvements, margins, d_trials = \
            count_successul_perturbations(dataset, depth, debug=True)
         data[(dataset, depth)] = improvements, margins
         trials += d_trials

         max_y_imp = max(max_y_imp, max(improvements))
         max_y_mar = max(max_y_mar, max(margins))

   plot_bins(axes[0], data[(Dataset.MNIST, 3)][index], data[(Dataset.FASHION, 3)][index])
   plot_bins(axes[1], data[(Dataset.MNIST, 6)][index], data[(Dataset.FASHION, 6)][index])

   if index == 0: axes[0].legend(loc="upper left")

   axes[0].set_ylabel(YLABELS[index])
   axes[0].set_xlabel("Layer")
   axes[1].set_xlabel("Layer")

   axes[0].set_title("Depth 3")
   axes[1].set_title("Depth 6")

   axes[0].set_xticks(range(3))
   axes[1].set_xticks(range(6))

   max_y = 1.1*(max_y_imp if index == 0 else max_y_mar)
   axes[0].set_ylim((0, max_y))
   axes[1].set_ylim((0, max_y))
   axes[1].set_yticks(())

   if index == 1:
      labels = axes[0].get_yticks()
      axes[0].set_yticklabels([f"{100*l:.0f}%" for l in labels])

   fig.suptitle(TITLES[index])

   output_folder = Path("./fig_wp_layers/")
   output_folder.mkdir(parents=True, exist_ok=True)
   letter = "a" if index == 0 else "b"
   filename = f"wp-layers_@{CONNECTIONS.value}_{PERCENT}percent.{letter}.png"
   plt.tight_layout()
   plt.savefig(output_folder / filename)
   plt.show()

def main():
   fig, axes = plt.subplots(2,2,gridspec_kw={"width_ratios": [1,1.618]})
   fig.set_size_inches(16,10)

   max_y_imp = 0
   max_y_mar = 0
   data = {}
   trials = 0

   for dataset in (Dataset.MNIST, Dataset.FASHION):
      for depth in (3, 6):
         improvements, margins, d_trials = \
            count_successul_perturbations(dataset, depth, debug=True)
         data[(dataset, depth)] = improvements, margins
         trials += d_trials

         max_y_imp = max(max_y_imp, max(improvements))
         max_y_mar = max(max_y_mar, max(margins))

   plot_bins(axes[0,0], data[(Dataset.MNIST, 3)][0], data[(Dataset.FASHION, 3)][0])
   plot_bins(axes[0,1], data[(Dataset.MNIST, 6)][0], data[(Dataset.FASHION, 6)][0])
   plot_bins(axes[1,0], data[(Dataset.MNIST, 3)][1], data[(Dataset.FASHION, 3)][1])
   plot_bins(axes[1,1], data[(Dataset.MNIST, 6)][1], data[(Dataset.FASHION, 6)][1])

   axes[0,0].legend(loc="upper left")

   axes[0,0].set_ylabel("Total count of improvements")
   axes[1,0].set_ylabel("Total $F_1$ score margin")
   axes[1,0].set_xlabel("Layer")
   axes[1,1].set_xlabel("Layer")

   axes[0,0].set_title("Improvements / depth 3")
   axes[0,1].set_title("Improvements / depth 6")
   axes[1,0].set_title("Margins / depth 3")
   axes[1,1].set_title("Margins / depth 6")

   axes[0,0].set_xticks(range(3))
   axes[1,0].set_xticks(range(3))
   axes[0,1].set_xticks(range(6))
   axes[1,1].set_xticks(range(6))

   max_y_imp *= 1.1; max_y_mar *= 1.1
   axes[0,0].set_ylim((0, max_y_imp))
   axes[0,1].set_ylim((0, max_y_imp))
   axes[1,0].set_ylim((0, max_y_mar))
   axes[1,1].set_ylim((0, max_y_mar))

   axes[0,1].set_yticks(())
   axes[1,1].set_yticks(())
   labels = axes[1,0].get_yticks()
   axes[1,0].set_yticklabels([f"{100*l:.0f}%" for l in labels])

   fig.suptitle("Weight perturbation improvements counted across respective layers")

   output_folder = Path("./fig_wp_layers/")
   output_folder.mkdir(parents=True, exist_ok=True)
   filename = f"wp-layers_@{CONNECTIONS.value}_{PERCENT}percent.png"
   plt.tight_layout()
   plt.savefig(output_folder / filename)
   plt.show()

   caption = (
      f"Total improvements in F1 score attributed to the respective layers perturbed. "
      f"The results compiled are the unweighted accumulation of {trials} total trials."
   )

   lines = [
      "% Generated with ``scripts/fig_wp_layers.py``",
      "\\begin{figure}[h]",
      "   \\centering",
      f"   \\includegraphics[width=\\textwidth]{{{filename}}}",
      f"   \\caption{{{caption}}}",
      f"   \\label{{fig:layers_@{CONNECTIONS.value}_{PERCENT}percent}}",
      "\\end{figure}",
   ]

   code = "\n".join(lines)
   print("```latex")
   print(code)
   print("```")

if __name__ == '__main__':
   #_ = main()
   plot_subfigures(0)
   plot_subfigures(1)

