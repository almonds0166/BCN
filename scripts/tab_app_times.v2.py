"""
Table of training times for Appendix C

All are averages. Split by shape, connections, dataset. SGD and WP.
"""

import sys; sys.path.append("../")
from pathlib import Path

from bcn import Results, Connections

from plotutils import ABBREVIATIONS

RESULTS_PATH = Path(input("Enter the location of all your *baseline* results\n> "))
WP_PATH = Path(input("Enter the location of all your *WP* results\n> "))
BATCH_SIZE = 64

def sgd_row(dataset, h, d, c, b, t):
   fname = f"results_{h}x{h}x{d}@{c}-{b}.{dataset}.b{BATCH_SIZE}.t{t}.pkl"
   file = RESULTS_PATH / fname

   if not file.exists(): return 0, 0, 0

   r = Results()
   r.load(file)

   t_train = sum(r.train_times)
   t_valid = sum(r.valid_times)
   t_total = t_train + t_valid

   return t_train, t_valid, 1

   return f"{b_} & {t} & {t_train:.0f} & {t_valid:.0f} & {t_total:.0f} \\\\"

def make_sgd_table():
   lines = [
      "% Generated from ``tab_app_times.v2.py``",
      "\\begin{table}",
      "\\centering",
      "\\begin{tabular}{c c c c c c c c}",
      (
         "{\\footnotesize Connections} & {\\footnotesize Dataset} & Size & Epochs & "
         "\\makecell{\\footnotesize Training \\\\[-1ex] \\footnotesize time (m)} & "
         "\\makecell{\\footnotesize Validation \\\\[-1ex] \\footnotesize time (m)} & "
         "\\makecell{\\footnotesize Total \\\\[-1ex] \\footnotesize time (m)} & Count \\\\"
      ),
      "\\hline",
   ]

   for c in (9, 25):
      flag_c = f"\\multirow{{8}}*{{1-to-{c}}}"
      for dataset in ("MNIST", "FASHION"):
         d_ = "MNIST" if dataset == "MNIST" else "Fashion"
         flag_d = f"\\multirow{{4}}*{{\\footnotesize {d_}}}"
         for h in (16, 30):
            for d in (3, 6):

               train = 0
               valid = 0
               count = 0
               for b in ABBREVIATIONS.keys():
                  for t in (1,2,3):
                     dt, dv, dc = sgd_row(dataset, h, d, c, b, t)
                     train += dt
                     valid += dv
                     count += dc

               total = train + valid
               train /= count
               valid /= count
               total /= count

               e = 40 if c == 25 else 100

               lines.append((
                  f"{flag_c} & {flag_d} & {h}x{h}x{d} & {e} & "
                  f"{train/60:0.1f} & {valid/60:0.1f} & {total/60:0.1f} & {count} \\\\"
               ))

               flag_c = ""
               flag_d = ""

      lines.append("\\hline")


   short_caption = f"Time to train various BCN models with SGD"
   caption = (
      f"Average time in minutes to train BCN models with "
      f"stochastic gradient descent."
   )
   lines.extend([
      "\\end{tabular}",
      f"\\caption[{short_caption}]{{{caption}}}",
      f"\\label{{tab:times-sgd}}",
      "\\end{table}",
      "",
   ])

   code = "\n".join(lines)

   print("```latex")
   print(code)
   print("```")

   return code

def wp_row(dataset, p, h, d, c, b, t):
   fname = f"results_{h}x{h}x{d}@{c}-{b}.{dataset}.b{BATCH_SIZE}.t{t}.pkl"
   file = WP_PATH / f"{p}percent" / fname

   if not file.exists(): return 0, 0, 0

   r = Results()
   r.load(file)

   t_train = sum(r.train_times[-r.step:])
   t_valid = sum(r.valid_times[-r.step:])
   t_total = t_train + t_valid

   return t_train, t_valid, 1

def make_wp_table():
   lines = [
      "% Generated from ``tab_app_times.v2.py``",
      "\\begin{table}",
      "\\centering",
      "\\begin{tabular}{c c c c c c}",
      (
         "Dataset & Size & "
         "\\makecell{Training \\\\ time (m)} & "
         "\\makecell{Validation \\\\ time (m)} & "
         "\\makecell{Total \\\\ time (m)} & Count \\\\"
      ),
      "\\hline",
   ]

   for c in (9,):
      for dataset in ("MNIST", "FASHION"):
         d_ = "MNIST" if dataset == "MNIST" else "Fashion-MNIST"
         flag_d = f"\\multirow{{4}}*{{{d_}}}"
         for h in (16, 30):
            for d in (3, 6):

               train = 0
               valid = 0
               count = 0
               for b in ABBREVIATIONS.keys():
                  for p in (2, 10):
                     for t in (1,2,3):
                        dt, dv, dc = wp_row(dataset, p, h, d, c, b, t)
                        train += dt
                        valid += dv
                        count += dc

               total = train + valid
               train /= count
               valid /= count
               total /= count

               lines.append((
                  f"{flag_d} & {h}x{h}x{d} & "
                  f"{train/60:0.1f} & {valid/60:0.1f} & {total/60:0.1f} & {count} \\\\"
               ))

               flag_d = ""

         lines.append("\\hline")

   short_caption = f"Time to train various BCN models with weight perturbation"
   caption = (
      f"Average time in minutes to train BCN models with "
      f"weight perturbation, all for 1-to-9 connections for 1000 steps."
   )
   lines.extend([
      "\\end{tabular}",
      f"\\caption[{short_caption}]{{{caption}}}",
      f"\\label{{tab:times-wp}}",
      "\\end{table}",
      "",
   ])

   code = "\n".join(lines)

   print("```latex")
   print(code)
   print("```")

   return code


def main():
   make_sgd_table()
   #make_wp_table()

if __name__ == "__main__":
   _ = main()