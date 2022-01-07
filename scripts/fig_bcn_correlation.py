"""
Horizontal - average output connectedness per neuron of input layer
Vertical - F1 score

One for MNIST, one for Fashion-MNIST
"""
import sys
sys.path.append("../")

import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from bcn import BCN, Connections, Results

from plotutils import (
   ABBREVIATIONS,
   BRANCHES,
   HIGH_CONTRAST,
   save_fig,
   keypad_connectedness,
)
BLUE, RED, YELLOW = HIGH_CONTRAST

BATCH_SIZE = 64
FOLDER_NAME = "fig_bcn_correlation"
DATASET = "MNIST"
RESULTS_PATH = Path(input("Enter the location of all your results\n> "))

FONT_SIZE = 22
plt.rcParams.update({'font.size': FONT_SIZE})

def average_oc_per_input_neuron(h, d, c, b):
   """OC is a function of h, w, d, c, b
   """
   model = BCN(h, d,
      connection=Connections(c),
      branches=BRANCHES[b]
   )
   oc = keypad_connectedness(model)
   first_layer = oc[0]
   print(h,d,c,b, first_layer)
   average = np.mean(first_layer)
   print(average)

   return average

def save_f1():
   f1 = {}
   for c in (9, 25):
      for h in (16, 30,):
         for d in (3, 6):
            for b, branches in BRANCHES.items():
               for t in (1,2,3):
                  bucket = (h, d, b)
                  fname = f"results_{h}x{h}x{d}@{c}-{b}.{DATASET}.b{BATCH_SIZE}.t{t}.pkl"
                  file = RESULTS_PATH / fname
                  if not file.exists(): continue
                  print(fname)
                  r = Results()
                  r.load(file)
                  best_f1 = max(r.f1_scores)
                  if bucket not in f1: f1[bucket] = []
                  f1[bucket].append(best_f1)

   print("len(f1) =", len(f1))
   pickle.dump(f1, open(f"./{FOLDER_NAME}/f1.{DATASET}.pkl", "wb"))

def save_cache():
   cache = {}
   for c in (9, 25):
      for h in (16, 30,):
         for d in (3, 6):
            for b, branches in BRANCHES.items():
               bucket = (h, d, b)
               cache[bucket] = average_oc_per_input_neuron(h, d, c, b)

   pickle.dump(cache, open(f"./{FOLDER_NAME}/cache.pkl", "wb"))

def save_data():
   """
   width, branches, oc, f1
   """
   cache = pickle.load(open(f"./{FOLDER_NAME}/cache.pkl", "rb"))
   f1 = pickle.load(open(f"./{FOLDER_NAME}/f1.{DATASET}.pkl", "rb"))

   data = []

   for c in (9, 25):
      for h in (16, 30,):
         for d in (3, 6):
            for b, branches in BRANCHES.items():
               bucket = (h, d, b)
               if bucket not in f1: continue

               oc = cache[bucket]
               for score in f1[bucket]:
                  type_ = "D" if b == "DirectOnly" else "B" # direct only vs. branches

                  datum = (h, type_, oc, score)
                  data.append(datum)

   pickle.dump(data, open(f"./{FOLDER_NAME}/data.{DATASET}.pkl", "wb"))               

def plot_data():
   data_m = pickle.load(open(f"./{FOLDER_NAME}/data.MNIST.pkl", "rb"))
   data_f = pickle.load(open(f"./{FOLDER_NAME}/data.FASHION.pkl", "rb"))
   data = data_m + data_f

   dir_oc = []
   dir_f1 = []
   bra_oc = []
   bra_f1 = []
   for h, t, o, s in data:
      if t == "D":
         dir_oc.append(o)
         dir_f1.append(s)
      else:
         bra_oc.append(o)
         bra_f1.append(s)

   u = set(dir_oc + bra_oc)
   print("len(u) =", len(u))
   N = len(dir_f1 + bra_f1)
   print("N =", N)

   x = np.array(dir_oc + bra_oc)
   y = np.array(dir_f1 + bra_f1)
   slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
   line_label = f"$r={r_value:+.2f}$ ($r^2={r_value**2:.2f}$)"
   coef = np.polyfit(x,y,1)
   line = np.poly1d(coef)

   fig, axes = plt.subplots(figsize=(12,9))
   ax = axes

   eps = 0.0
   ax.scatter(np.array(dir_oc)-eps, dir_f1,marker="s", color=BLUE, label="Direct only connections")
   ax.scatter(np.array(bra_oc)+eps, bra_f1, marker="o", color=RED, label="Branching connections")
   x_ = np.array([min(x)-0.2, max(x)+0.2])
   ax.plot(x_, line(x_), linestyle="--", color=YELLOW, label=line_label, linewidth=3)

   ax.set_ylim((0.4-0.02, 1.02))
   ax.set_xlim((-0.2, 10.2))
   ax.set_xticks(range(0,11))
   ax.set_ylabel("$F_1$ score")
   ax.set_xlabel("Average output connectedness\nper neuron in input layer")
   ax.set_title((
      f"Correlation of output connectedness and $F_1$ score\n"
      f"across $N={N}$ BCN results"
   ))
   ax.legend(loc="lower right")

   plt.tight_layout()
   save_fig(plt, f"./{FOLDER_NAME}", "regression.png", True)


if __name__ == "__main__":
   print("Useful functions:")
   print("* save_cache()")
   print("* save_f1()")
   print("* save_data()")
   print("* plot_data()")