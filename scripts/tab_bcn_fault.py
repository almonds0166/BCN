
"""
Tabulate the maximum F1 scores for each of the eight types of networks investigate in the thesis,
*before and after the fault*, and emphasize the differences, too.

This is for chapter 3 of my thesis.

The given path for the *WP* results should contain the following structure:

   ./{p}percent/...

where the inner-most folder contains all the pkl & pt files, and ``p`` represents the percentage of
fault, e.g. 2 or 10 for 2% or 10% respectively. Remember that the file that had WP ran on it have
trial names like "t1" or "t3", but the ones that had SGD ran on them have trial names like "t1o"
and "t2o", where the "o" kind of represents an origin or baseline or control group.

Ideas I have in mind for improvement:
- You know what? I might just abandon these tables and go with more visually appealing figures...
"""

import sys; sys.path.append("../")
from pathlib import Path
import re

from bcn import Results, Dataset, Connections

from plotutils import ABBREVIATIONS

RESULTS_PATH = Path(input("Enter the location of your *BCN* results\n> "))
WP_PATH = Path(input("Enter the location of your *WP* results\n> "))

DATASET = Dataset.FASHION
CONNECTIONS = Connections.ONE_TO_9
PERCENT = 10

# get subfolder with only the desired results
WP_PATH /= f"{PERCENT}percent"

def mean(ell, default=None):
   return sum(ell) / len(ell) if ell else default

def main():

   # get the BEFORE fault data
   # these are the results found in RESULTS_PATH
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

      # get the BEST data point
      vl = min(r.valid_losses)
      ac = max(r.accuracies)
      f1 = max(r.f1_scores)

      before_data[bucket]["vl"].append(vl)
      before_data[bucket]["ac"].append(ac)
      before_data[bucket]["f1"].append(f1)

   # get the results immediately after fault
   # these are the results found in WP_PATH
   after_data = {}

   for file in WP_PATH.iterdir():
      if not file.name.startswith("results_") or file.suffix != ".pkl": continue
      if not f".{DATASET.name}." in file.stem: continue
      if not f"@{CONNECTIONS.value}-" in file.stem: continue
      if file.stem.endswith("o"): continue # just skip the SGD ones

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

      if bucket not in after_data:
         after_data[bucket] = {
            "vl": [],
            "ac": [],
            "f1": [],
         }

      # get the FIRST data point
      index = -(r.step + 1)
      vl = r.valid_losses[index]
      ac = r.accuracies[index]
      f1 = r.f1_scores[index]

      after_data[bucket]["vl"].append(vl)
      after_data[bucket]["ac"].append(ac)
      after_data[bucket]["f1"].append(f1)

   # integrate the data into one nice table!
   lines = [
      "% Generated with ``scripts/tab_bcn_fault.py``",
      "\\begin{table}",
      "\\centering",
      "\\begin{tabular}{ " + ("c " * 6) + "}",
      "\\hline",
      (
         "\\multicolumn{2}{ c }{Model} & "
         "\\multicolumn{4}{ c }{F1 score} \\\\"
      ),
      "\\hline",
      (
         "Shape & Branches & "
         "Before fault & "
         "After fault & "
         "Abs. diff. & "
         "Rel. diff. \\\\"
      ),
      "\\hline",
   ]

   # minimum number of trials
   trials = float("inf")

   for w in (16, 30):
      h = w
      for d in (3, 6):
         for b in ABBREVIATIONS.keys(): # dict keys are ordered starting in ~3.7
            bucket = (h, w, d, b)

            if not bucket in before_data: continue
            if not bucket in after_data: continue

            trials = min(trials, len(before_data[bucket]), len(after_data[bucket]))

            before_vl = mean(before_data[bucket]["vl"])
            before_ac = mean(before_data[bucket]["ac"])
            before_f1 = mean(before_data[bucket]["f1"])

            after_vl = mean(after_data[bucket]["vl"])
            after_ac = mean(after_data[bucket]["ac"])
            after_f1 = mean(after_data[bucket]["f1"])

            abs_diff_vl = after_vl - before_vl
            abs_diff_ac = after_ac - before_ac
            abs_diff_f1 = after_f1 - before_f1

            rel_diff_vl = abs_diff_vl / before_vl
            rel_diff_ac = abs_diff_ac / before_ac
            rel_diff_f1 = abs_diff_f1 / before_f1

            lines.append((
               f"{h}x{w}x{d} & "
               f"{ABBREVIATIONS[b]} & "
               #f"{before_vl:.3f} & "
               #f"{100*before_ac:.1f}\\% & "
               f"{100*before_f1:.1f}\\% & "
               #f"{after_vl:.3f} & "
               #f"{100*after_ac:.1f}\\% & "
               f"{100*after_f1:.1f}\\% & "
               #f"{abs_diff_vl:+.3f} & "
               #f"{100*abs_diff_ac:+.1f}\\% & "
               f"{100*abs_diff_f1:+.1f}\\% & "
               #f"{rel_diff_vl:+.2f} & "
               #f"{rel_diff_ac:+.2f} & "
               f"{rel_diff_f1:+.2f} \\\\"
            ))
         lines.append("\\hline")

   lines.extend([
      #"\\hline",
      "\\end{tabular}",
      (
         f"\\caption{{Drop in performance with {PERCENT}\\% applied fault for {DATASET.value} "
         f"at 1-to-{CONNECTIONS.value} connections (average of {trials} trials).}}"
      ),
      f"\\label{{table:ch3:fault_{DATASET.name.lower()}"
      f"@{CONNECTIONS.value}_{PERCENT}percent}}",
      "\\end{table}",
   ])

   code = "\n".join(lines)
   print("```latex")
   print(code)
   print("```")

   return code

if __name__ == '__main__':
   _ = main()

