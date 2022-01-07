"""
Table of training times for Appendix C
"""

import sys; sys.path.append("../")
from pathlib import Path

from bcn import Results, Connections

from plotutils import ABBREVIATIONS

RESULTS_PATH = Path(input("Enter the location of all your *baseline* results\n> "))
WP_PATH = Path(input("Enter the location of all your *WP* results\n> "))
BATCH_SIZE = 64

DATASETS = ("MNIST", "FASHION")
HEIGHTS = (16, 30)
DEPTHS = (3, 6)
CONNECTIONS = (9, 25)
PERCENTS = (2, 10)
BRANCHES = ABBREVIATIONS.keys()
TRIALS = (1, 2, 3)

global accumulator, total
accumulator = [0, 0, 0, 0]
total = {"sgd": [0, 0, 0], "wp": [0, 0, 0]}

def sgd_row(dataset, h, d, c, b, t):
   fname = f"results_{h}x{h}x{d}@{c}-{b}.{dataset}.b{BATCH_SIZE}.t{t}.pkl"
   file = RESULTS_PATH / fname
   r = Results()
   r.load(file)

   t_train = sum(r.train_times)
   t_valid = sum(r.valid_times)
   t_total = t_train + t_valid

   # minutes instead of seconds
   t_train /= 60
   t_valid /= 60
   t_total /= 60

   global accumulator, total
   accumulator[0] += t_train
   accumulator[1] += t_valid
   accumulator[2] += t_total
   accumulator[3] += 1
   total["sgd"][0] += t_train
   total["sgd"][1] += t_valid
   total["sgd"][2] += 1

   b_ = f"\\multirow{{{len(TRIALS)}}}*{{{ABBREVIATIONS[b]}}}" if t == 1 else ""

   return f"{b_} & {t} & {t_train:.0f} & {t_valid:.0f} & {t_total:.0f} \\\\"

def make_sgd_table():
   lines = [
      "% Generated from ``tab_app_times.py``"
   ]

   for c in CONNECTIONS:
      for h in HEIGHTS:
         for d in DEPTHS:
            for dataset in DATASETS:
               lines.extend([
                  "\\begin{table}",
                  "\\centering",
                  "\\begin{tabular}{c c c c c}",
                  (
                     "Branches & Trial & "
                     "\\makecell{Training \\\\ time (m)} & "
                     "\\makecell{Validation \\\\ time (m)} & "
                     "\\makecell{Total \\\\ time (m)} \\\\"
                  ),
                  "\\hline",
               ])

               global accumulator
               accumulator = [0, 0, 0, 0]

               for b in BRANCHES:
                  for t in TRIALS:
                     line = sgd_row(dataset, h, d, c, b, t)
                     lines.append(line)
                  lines.append("\\hline")

               t_train, t_valid, t_total, count = accumulator
               a_train, a_valid, a_total, _ = [a/count for a in accumulator]

               lines.extend([
                  (
                     f"\\multicolumn{{2}}{{c}}{{Average}} & "
                     f"{a_train:.0f} & {a_valid:.0f} & {a_total:.0f} \\\\"
                  ),
                  f"\\hline",
                  #(
                  #   f"\\multicolumn{{2}}{{c}}{{Sum}} & "
                  #   f"{t_train:.0f} & {t_valid:.0f} & {t_total:.0f} \\\\"
                  #),
                  #f"\\hline",
               ])
               
               d_ = dataset if dataset == "MNIST" else "Fashion-MNIST"

               short_caption = f"Time to train {h}x{h}x{d} 1-to-{c} models on {d_} with SGD"
               caption = (
                  f"Time in minutes to train {h}x{h}x{d} 1-to-{c} models on {d_} with "
                  f"stochastic gradient descent (x epochs each)."
               )
               lines.extend([
                  "\\end{tabular}",
                  f"\\caption[{short_caption}]{{{caption}}}",
                  f"\\label{{tab:times-sgd-{h}x{h}x{d}@{c}.{dataset}}}",
                  "\\end{table}",
                  "",
               ])

   code = "\n".join(lines)

   #print("```latex")
   #print(code)
   #print("```")

   return code

def wp_row(dataset, p, h, d, c, b, t):
   fname = f"results_{h}x{h}x{d}@{c}-{b}.{dataset}.b{BATCH_SIZE}.t{t}.pkl"
   file = WP_PATH / f"{p}percent" / fname

   if not file.exists(): return ""

   r = Results()
   r.load(file)

   t_train = sum(r.train_times[-r.step:])
   t_valid = sum(r.valid_times[-r.step:])
   t_total = t_train + t_valid

   # minutes instead of seconds
   t_train /= 60
   t_valid /= 60
   t_total /= 60

   global accumulator, total
   accumulator[0] += t_train
   accumulator[1] += t_valid
   accumulator[2] += t_total
   accumulator[3] += 1
   total["wp"][0] += t_train
   total["wp"][1] += t_valid
   total["wp"][2] += 1

   b_ = f"\\multirow{{{len(TRIALS)}}}*{{{ABBREVIATIONS[b]}}}" if t == 1 else ""

   return f"{b_} & {t} & {t_train:.0f} & {t_valid:.0f} & {t_total:.0f} \\\\"

def make_wp_table():
   lines = []

   for p in PERCENTS:
      for c in CONNECTIONS:
         for h in HEIGHTS:
            for d in DEPTHS:
               for dataset in DATASETS:
                  lines.extend([
                     "\\begin{table}",
                     "\\centering",
                     "\\begin{tabular}{c c c c c}",
                     (
                        "Branches & Trial & "
                        "\\makecell{Training \\\\ time (m)} & "
                        "\\makecell{Validation \\\\ time (m)} & "
                        "\\makecell{Total \\\\ time (m)} \\\\"
                     ),
                     "\\hline",
                  ])

                  global accumulator
                  accumulator = [0, 0, 0, 0]

                  for b in BRANCHES:
                     for t in TRIALS:
                        line = wp_row(dataset, p, h, d, c, b, t)
                        lines.append(line)
                     lines.append("\\hline")

                  t_train, t_valid, t_total, count = accumulator; count = max(1, count)
                  a_train, a_valid, a_total, _ = [a/count for a in accumulator]

                  lines.extend([
                     (
                        f"\\multicolumn{{2}}{{c}}{{Average}} & "
                        f"{a_train:.0f} & {a_valid:.0f} & {a_total:.0f} \\\\"
                     ),
                     f"\\hline",
                     #(
                     #   f"\\multicolumn{{2}}{{c}}{{Sum}} & "
                     #   f"{t_train:.0f} & {t_valid:.0f} & {t_total:.0f} \\\\"
                     #),
                     #f"\\hline",
                  ])

                  d_ = dataset if dataset == "MNIST" else "Fashion-MNIST"

                  short_caption = (
                     f"Time to train {h}x{h}x{d} 1-to-{c} models on {d_} after {p}\\% fault "
                     f"with WP"
                  )
                  caption = (
                     f"Time in minutes to train {h}x{h}x{d} 1-to-{c} models on {d_} "
                     f"after {p}\\% applied fault with weight perturbation (1000 steps each)."
                  )
                  lines.extend([
                     "\\end{tabular}",
                     f"\\caption[{short_caption}]{{{caption}}}",
                     f"\\label{{tab:times-wp-{h}x{h}x{d}@{c}.{dataset}_{p}percent}}",
                     "\\end{table}",
                     "",
                  ])

   code = "\n".join(lines)
   code = code.replace("\n\n\n\\hline\n", "")

   return code

def make_cumulative_table():
   global total

   sgd_train, sgd_valid, sgd_count = total["sgd"]
   sgd_total = sgd_train + sgd_valid

   wp_train, wp_valid, wp_count = total["wp"]
   wp_total = wp_train + wp_valid

   short_caption = "Total thesis training times"
   caption = (
      "Total training times in hours across the main SGD and WP processes "
      "spotlighted in this thesis."
   )

   count = sgd_count + wp_count
   time = sgd_total + wp_total

   lines = [
      "\\begin{table}",
      "\\centering",
      "\\begin{tabular}{c | c c}",
      "Process & Count & Total time (h) \\\\",
      f"SGD & {sgd_count} & {sgd_total/60:.0f} \\\\",
      f"WP & {wp_count} & {wp_total/60:.0f} \\\\",
      f"Sum & {count} & {time/60:.0f} \\\\",
      "\\end{tabular}",
      f"\\caption[{short_caption}]{{{caption}}}",
      f"\\label{{tab:times-total}}",
      "\\end{table}",
      "",
   ]

   code = "\n".join(lines)

   return code

if __name__ == "__main__":
   code_sgd = make_sgd_table()
   code_wp = make_wp_table()
   code_all = make_cumulative_table()

   #print(code_all)
   #print(code_sgd)
   print(code_wp)