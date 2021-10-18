
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

from bcn import Results, Dataset, Connections

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

FONT_SIZE = 14
plt.rcParams.update({'font.size': FONT_SIZE})

def mean(ell, default=None):
   return sum(ell) / len(ell) if ell else default

def count_successul_perturbations(dataset, depth, debug=False):
   """Return a list of all tallied WP improvement with respect to each layer.

   Uses the WP_PATH as defined in the global frame above.
   """
   improvements = [0] * depth

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

      if debug: print(file.name)

      m = re.search(rf"@{CONNECTIONS.value}-([\w\.\(\)0-9]+).{dataset.name}.", file.stem)
      b = m.group(1)

      r = Results()
      r.load(file)

      N = len(r.wp_layers)
      for i, (l, f1) in enumerate(zip(r.wp_layers, r.f1_scores[-(N+1):])):
         next_f1 = r.f1_scores[i+1]
         improvements[l] += (next_f1 > f1) # record all improvements!

   return improvements

def main():
   fig, axes = plt.subplots(2,2)
   fig.set_size_inches(16,10)
   axes = axes.flatten()

   max_y = 0

   axes_iterator = iter(axes)
   for dataset in (Dataset.MNIST, Dataset.FASHION):
      for depth in (3, 6):
         ax = next(axes_iterator)

         bins = count_successul_perturbations(dataset, depth, debug=True)
         max_y = max(max_y, max(bins))

         ax.bar(range(depth), bins)

         ax.set_xlabel("Layer")
         ax.set_ylabel("Total count of improvements")
         ax.set_title(f"Depth {depth}, {dataset.value}")
         ax.set_xticks(range(depth))

   max_y = round(1.1*max_y)
   for ax in axes:
      ax.set_ylim((0, max_y))

   fig.suptitle("Weight perturbation improvements counted across respective layers")

   output_folder = Path("./fig_wp_layers/")
   output_folder.mkdir(parents=True, exist_ok=True)
   filename = f"wp-layers_@{CONNECTIONS.value}_{PERCENT}percent.png"
   plt.tight_layout()
   plt.savefig(output_folder / filename)
   plt.show()

   caption = (
      f"Total improvements in F1 score attributed to the respective layers perturbed. "
      f"Data shown for MNIST and Fashion-MNIST datasets."
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
   _ = main()

