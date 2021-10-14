
"""
Tabulate the minimum validation losses and maximum scores for each of the eight types of networks
investigate in the thesis.

This is for chapter 3 of my thesis.

Ideas I have in mind for improvement:
- No need to repeat model shape so many times
- Emphasize the top three f1 scores in each shape section
"""

import sys; sys.path.append("../")
from pathlib import Path
import re

from bcn import Results, Dataset, Connections

RESULTS_PATH = Path(input("Enter the location of all your results\n> "))

DATASET = Dataset.MNIST
CONNECTIONS = Connections.ONE_TO_9

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

def mean(ell, default=None):
   return sum(ell) / len(ell) if ell else default

def main():
   lines = [
      "% Generated with ``scripts/tab_bcn_scores.py``",
      "\\begin{table}",
      "\\centering",
      "\\begin{tabular}{ c c c c c }",
      "\\hline",
      "\\multicolumn{2}{ c }{Model} & \\multicolumn{3}{ c }{Metrics} \\\\",
      "\\hline",
      "Shape & Branches & Loss & Accuracy & F1 score \\\\",
      "\\hline",
   ]

   data = {}

   epochs = float("inf") # min number of epochs each
   trials = float("inf") # min number of trials each

   for file in RESULTS_PATH.iterdir():
      if not file.name.startswith("results_") or file.suffix != ".pkl": continue
      if not f".{DATASET.name}." in file.stem: continue
      if not f"@{CONNECTIONS.value}-" in file.stem: continue

      print(file.name)

      r = Results()
      r.load(file)

      epochs = min(epochs, r.epoch)

      m = re.search(r"([0-9]+)x([0-9]+)x([0-9]+)", file.stem)
      h = int(m.group(1))
      w = int(m.group(2))
      d = int(m.group(3))

      m = re.search(rf"@{CONNECTIONS.value}-([\w\.\(\)0-9]+).{DATASET.name}.", file.stem)
      b = m.group(1)

      bucket = (h, w, d, b)

      if bucket not in data:
         data[bucket] = {
            "vl": [],
            "ac": [],
            "f1": [],
         }

      vl = min(r.valid_losses)
      ac = max(r.accuracies)
      f1 = max(r.f1_scores)

      data[bucket]["vl"].append(vl)
      data[bucket]["ac"].append(ac)
      data[bucket]["f1"].append(f1)

   #return data

   for w in (16, 30):
      h = w
      for d in (3, 6):
         for b in BRANCH_NAMES.keys(): # dict keys are ordered starting in ~3.7
            bucket = (h, w, d, b)

            trials = min(trials, len(data[bucket]))

            vl = mean(data[bucket]["vl"])
            ac = mean(data[bucket]["ac"])
            f1 = mean(data[bucket]["f1"])

            lines.append((
               f"{h}x{w}x{d} & {BRANCH_NAMES[b]} & "
               f"{vl:.3f} & {100*ac:.1f}\\% & {100*f1:.1f}\\% \\\\"
            ))
         lines.append("\\hline")

   lines.extend([
      #"\\hline",
      "\\end{tabular}",
      (
         f"\\caption{{Results for {DATASET.value} at 1-to-{CONNECTIONS.value} connections "
         f"({epochs} epochs, {trials} trials each).}}"
      ),
      f"\\label{{table:ch3:scores_{DATASET.name.lower()}@{CONNECTIONS.value}}}",
      "\\end{table}",
   ])

   code = "\n".join(lines)
   print("```latex")
   print(code)
   print("```")

   return code

if __name__ == '__main__':
   _ = main()

