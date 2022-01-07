
import sys
sys.path.append("../")

from pathlib import Path
import re

import numpy as np

from bcn import Results

WP_FROM_SCRATCH_LOC = Path(input("Enter the location of all your *WP from scratch* results\n> "))

EPS = (1, 5, 10, 20)

def mean(ell, default=None):
   return sum(ell) / len(ell) if ell else default

def main():
   lines = [
      "% Generated with ``scripts/tab_wp_scratch.py``",
      "\\begin{table}",
      "\\centering",
      "\\begin{tabular}{ " + ("c " * 6) + "}",
      "\\hline",
      "& & & & \\multicolumn{2}{c}{Improvements} \\\\",
      (
         "$\\varepsilon$ & Initial $F_1$ & Top $F_1$ & Difference & "
         "Train loss & $F_1$ score \\\\"
      ),
      "\\hline",
   ]

   for eps in EPS:
      loc = WP_FROM_SCRATCH_LOC / f"{eps}percent"

      data = {
         "initial": [],
         "final": [],
         "change": [],
         "f1_improvements": [],
         "tl_improvements": [],
      }

      for file in loc.iterdir():
         if (
            not file.name.startswith("results_")
            or file.suffix != ".pkl"
         ):
            continue

         print(file.name)

         r = Results()
         r.load(file)

         f1_0 = r.f1_scores[0]
         f1_f = max(r.f1_scores)
         change = f1_f - f1_0

         data["initial"].append(f1_0)
         data["final"].append(f1_f)
         data["change"].append(change)

         N = len(r.wp_layers)
         improvements = 0
         for i, f1 in enumerate(r.f1_scores[:N-1]):
            next_f1 = r.f1_scores[i+1]
            improvements += (next_f1 > f1)

         data["f1_improvements"].append(improvements)

         N = len(r.train_losses)
         improvements = 0
         for i, tl in enumerate(r.train_losses[:N-1]):
            next_tl = r.train_losses[i+1]
            improvements += (next_tl < tl)

         data["tl_improvements"].append(improvements)

      initial = 100*mean(data["initial"])
      final   = 100*mean(data["final"])
      change  = 100*mean(data["change"])
      tl_count = mean(data["tl_improvements"])
      f1_count = mean(data["f1_improvements"])

      line = (
         f"{eps/100:.02f} & {initial:.2f}\\% & {final:.2f}\\% & {change:+.2f}\\% & "
         f"{tl_count:.1f} & {f1_count:.1f} \\\\"
      )
      lines.append(line)




   lines.extend([
      "\\hline",
      "\\end{tabular}",
      "\\caption[]{}",
      "\\label{tab:scratch}",
      "\\end{table}",
   ])

   code = "\n".join(lines)

   print("```latex")
   print(code)
   print("```")

   return code

if __name__ == "__main__":
   _ = main()