
"""
Tabulate the F1 scores before and after WP.

This is for chapter 4 of my thesis.

The given path for the *WP* results should contain the following structure:

   ./{p}percent/...

where the inner-most folder contains all the pkl & pt files, and ``p`` represents the percentage of
fault, e.g. 2 or 10 for 2% or 10% respectively. Remember that the file that had WP ran on it have
trial names like "t1" or "t3", but the ones that had SGD ran on them have trial names like "t1o"
and "t2o", where the "o" kind of represents an origin or baseline or control group.

Ideas I have in mind for improvement:
- None
"""

import sys; sys.path.append("../")
from pathlib import Path
import re

from bcn import Results, Dataset, Connections

from plotutils import ABBREVIATIONS

RESULTS_PATH = Path(input("Enter the location of all your *baseline* results\n> "))
WP_PATH = Path(input("Enter the location of your *WP* results\n> "))

DATASET = Dataset.MNIST
CONNECTIONS = Connections.ONE_TO_9
PERCENT = 10

# get subfolder with only the desired results
WP_PATH /= f"{PERCENT}percent"

def mean(ell, default=None):
   return sum(ell) / len(ell) if ell else default

def main():

   # get the BEFORE and AFTER WP data
   # these are the results found in WP_PATH
   wp_data = {}

   for file in WP_PATH.iterdir():
      if not file.name.startswith("results_") or file.suffix != ".pkl": continue
      if not f".{DATASET.name}." in file.stem: continue
      if not f"@{CONNECTIONS.value}-" in file.stem: continue
      if file.stem.endswith("o"): continue # skip the SGD ones

      print(file.name)

      r = Results()
      r.load(file)

      m = re.search(r"([0-9]+)x([0-9]+)x([0-9]+)", file.stem)
      h = int(m.group(1))
      w = int(m.group(2))
      d = int(m.group(3))

      m = re.search(rf"@{CONNECTIONS.value}-([\w\.\(\)0-9]+).{DATASET.name}.", file.stem)
      b = m.group(1)

      bucket = (h, w, d, b)

      if bucket not in wp_data:
         wp_data[bucket] = {
            "before": {"vl": [], "ac": [], "f1": []},
            "after": {"vl": [], "ac": [], "f1": []},
         }

      # get the FIRST data point
      index = -(r.step + 1)
      vl = r.valid_losses[index]
      ac = r.accuracies[index]
      f1 = r.f1_scores[index]

      wp_data[bucket]["before"]["vl"].append(vl)
      wp_data[bucket]["before"]["ac"].append(ac)
      wp_data[bucket]["before"]["f1"].append(f1)

      # get the last data point
      vl = r.valid_losses[-1]
      ac = r.accuracies[-1]
      f1 = r.f1_scores[-1]

      wp_data[bucket]["after"]["vl"].append(vl)
      wp_data[bucket]["after"]["ac"].append(ac)
      wp_data[bucket]["after"]["f1"].append(f1)

   # get the results for SGD recovery
   # these are the results found in WP_PATH
   sgd_data = {}

   for file in WP_PATH.iterdir():
      if not file.name.startswith("results_") or file.suffix != ".pkl": continue
      if not f".{DATASET.name}." in file.stem: continue
      if not f"@{CONNECTIONS.value}-" in file.stem: continue
      if not file.stem.endswith("o"): continue # skip the WP ones

      print(file.name)

      r = Results()
      r.load(file)

      m = re.search(r"([0-9]+)x([0-9]+)x([0-9]+)", file.stem)
      h = int(m.group(1))
      w = int(m.group(2))
      d = int(m.group(3))

      m = re.search(rf"@{CONNECTIONS.value}-([\w\.\(\)0-9]+).{DATASET.name}.", file.stem)
      b = m.group(1)

      bucket = (h, w, d, b)

      if bucket not in sgd_data:
         sgd_data[bucket] = {
            "vl": [],
            "ac": [],
            "f1": [],
         }

      # get the final data point
      vl = r.valid_losses[-1]
      ac = r.accuracies[-1]
      f1 = r.f1_scores[-1]

      sgd_data[bucket]["vl"].append(vl)
      sgd_data[bucket]["ac"].append(ac)
      sgd_data[bucket]["f1"].append(f1)

   # get the BEFORE FAULT results...
   before_data = {}
   for file in RESULTS_PATH.iterdir():
      if not file.name.startswith("results_") or file.suffix != ".pkl": continue
      if not f".{DATASET.name}." in file.stem: continue
      if not f"@{CONNECTIONS.value}-" in file.stem: continue

      print(file.name)

      r = Results()
      r.load(file)

      m = re.search(r"([0-9]+)x([0-9]+)x([0-9]+)", file.stem)
      h = int(m.group(1))
      w = int(m.group(2))
      d = int(m.group(3))

      m = re.search(rf"@{CONNECTIONS.value}-([\w\.\(\)0-9]+).{DATASET.name}.", file.stem)
      b = m.group(1)

      bucket = (h, w, d, b)

      if bucket not in before_data:
         before_data[bucket] = {
            "vl": [],
            "ac": [],
            "f1": [],
         }

      # get the BEFORE FAULT data point
      vl = r.valid_losses[r.best]
      ac = r.accuracies[r.best]
      f1 = r.f1_scores[r.best]

      before_data[bucket]["vl"].append(vl)
      before_data[bucket]["ac"].append(ac)
      before_data[bucket]["f1"].append(f1)

   # integrate the data into one nice table!
   lines = [
      "% Generated with ``scripts/tab_wp_scores.py``",
      "\\begin{table}",
      "\\centering",
      "\\begin{tabular}{ " + ("c " * 8) + "}",
      "\\hline",
      (
         "Model & "
         "{\\footnotesize Before fault} & "
         "{\\footnotesize After fault} & "
         "SGD & "
         "WP & "
         "$\\Delta_\\mathrm{SGD}$ & "
         "$\\Delta_\\mathrm{WP}$ & "
         "{\\footnotesize $\\Delta_\\mathrm{WP}/\\Delta_\\mathrm{SGD}$} \\\\"
      ),
      "\\hline",
   ]

   # minimum number of trials
   trials = float("inf")

   for w in (16, 30):
      h = w
      for d in (3, 6):
         lines.append((
            "\\multicolumn{8}{l}{"
            f"{h}x{w}x{d}"
            "} \\\\"
         ))
         lines.append("\\hline")

         for b in ABBREVIATIONS.keys(): # dict keys are ordered starting in ~3.7
            bucket = (h, w, d, b)

            if bucket not in wp_data: continue # !!

            trials = min(trials,
               len(wp_data[bucket]["before"]),
               len(wp_data[bucket]["after"]),
               len(sgd_data[bucket])
            )

            before_fault_vl = mean(before_data[bucket]["vl"])
            before_fault_ac = mean(before_data[bucket]["ac"])
            before_fault_f1 = mean(before_data[bucket]["f1"])

            before_wp_vl = mean(wp_data[bucket]["before"]["vl"])
            before_wp_ac = mean(wp_data[bucket]["before"]["ac"])
            before_wp_f1 = mean(wp_data[bucket]["before"]["f1"])

            after_wp_vl = mean(wp_data[bucket]["after"]["vl"])
            after_wp_ac = mean(wp_data[bucket]["after"]["ac"])
            after_wp_f1 = mean(wp_data[bucket]["after"]["f1"])

            sgd_vl = mean(sgd_data[bucket]["vl"])
            sgd_ac = mean(sgd_data[bucket]["ac"])
            sgd_f1 = mean(sgd_data[bucket]["f1"])

            d_wp_vl = after_wp_vl - before_wp_vl
            d_wp_ac = after_wp_ac - before_wp_ac
            d_wp_f1 = after_wp_f1 - before_wp_f1

            d_sgd_vl = sgd_vl - before_wp_vl
            d_sgd_ac = sgd_ac - before_wp_ac
            d_sgd_f1 = sgd_f1 - before_wp_f1

            relative_vl = d_wp_vl / d_sgd_vl
            relative_ac = d_wp_ac / d_sgd_ac
            relative_f1 = d_wp_f1 / d_sgd_f1

            lines.append((
               #f"{h}x{w}x{d} & "
               f"{ABBREVIATIONS[b]} & "
               f"{100*before_fault_f1:.1f}\\% & "
               f"{100*before_wp_f1:.1f}\\% & "
               f"{100*sgd_f1:.1f}\\% & "
               f"{100*after_wp_f1:.1f}\\% & "
               f"{100*d_sgd_f1:+.1f}\\% &"
               f"{100*d_wp_f1:+.1f}\\% & "
               f"{relative_f1:.2f} \\\\"
            ))
         lines.append("\\hline")

   lines.extend([
      #"\\hline",
      "\\end{tabular}",
      (
         f"\\caption{{Recovery in performance from weight perturbation "
         f"after {PERCENT}\\% applied fault. "
         f"Results for {DATASET.value} at 1-to-{CONNECTIONS.value} connections, average of "
         f"{trials} trials.}}"
      ),
      f"\\label{{table:ch4:scores_{DATASET.name.lower()}"
      f"@{CONNECTIONS.value}}}_{PERCENT}percent",
      "\\end{table}",
   ])

   code = "\n".join(lines)
   print("```latex")
   print(code)
   print("```")

   return code

if __name__ == '__main__':
   _ = main()

