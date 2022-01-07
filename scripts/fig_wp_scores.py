
import sys; sys.path.append("../")
from pathlib import Path
import re

from matplotlib import pyplot as plt
import numpy as np

from bcn import Results, Dataset, Connections
#from bcn.branches import DirectOnly
#from bcn.branches.uniform import (NearestNeighbor, NearestNeighborOnly,
#                                  NextToNN, NextToNNOnly)
#from bcn.branches.informed import Kappa, IndirectOnly

from plotutils import (
   ABBREVIATIONS, ABBREVIATIONS_SAFE, save_fig,
   HIGH_CONTRAST
)

BATCH_SIZE = 64
PERCENT = 10

DATASET = (Dataset.MNIST,)
CONNECTIONS = (Connections.ONE_TO_9,)
BRANCHES = ("DirectOnly",)
HEIGHT = (16,)
DEPTH = (3,)
TRIAL = (2,)

WP_PATH = Path(input("Enter the location of all your *WP* results\n> "))
WP_PATH /= f"{PERCENT}percent" # get subfolder with only the desired results

# plot config
BLUE, RED, YELLOW = HIGH_CONTRAST
LW = 3
FONT_SIZE = 20
plt.rcParams.update({'font.size': FONT_SIZE})

def main():
   for height in HEIGHT:
      for depth in DEPTH:
         for dataset in DATASET:
            for connections in CONNECTIONS:
               for branches in BRANCHES:
                  for trial in TRIAL:
                     make_plot(height, depth, dataset, connections, branches, trial)

def make_plot(height, depth, dataset, connections, branches, trial, *, show=False):
   h = height; w = h; d = depth
   b = ABBREVIATIONS[branches]
   safe_b = ABBREVIATIONS_SAFE[branches]
   c = connections.value
   fname_wp = (
      f"results_{h}x{w}x{d}@{c}-"
      f"{branches}.{dataset.name}.b{BATCH_SIZE}.t{trial}.pkl"
   )
   fname_sgd = (
      f"results_{h}x{w}x{d}@{c}-"
      f"{branches}.{dataset.name}.b{BATCH_SIZE}.t{trial}o.pkl"
   )
   loc_wp = WP_PATH / fname_wp
   loc_sgd = WP_PATH / fname_sgd

   assert loc_wp.exists(), f"wp file {fname_wp!r} must exist"
   assert loc_sgd.exists(), f"sgd file {fname_sgd!r} must exist"

   fig, axes = plt.subplots(
      2, 2,
      gridspec_kw={"width_ratios": [1, 1.618]},
      figsize=(16,9),
   )

   # baseline results
   r = Results()
   r.load(loc_wp)
   wp_steps = r.step

   tl = r.train_losses[:100]
   f1 = r.f1_scores[:101]

   f1_before_fault = f1[-1]

   axes[0,0].plot(tl, color=BLUE, linewidth=LW, label="SGD")
   axes[1,0].plot(f1, color=BLUE, linewidth=LW)

   # wp recovery
   wp_tl = r.train_losses[100:]
   wp_f1 = r.f1_scores[101:]

   f1_after_fault = wp_f1[0]
   f1_after_recovery = wp_f1[-1]

   axes[0,1].plot(wp_tl, color=RED, linewidth=LW, label="WP")
   axes[1,1].plot(wp_f1, color=RED, linewidth=LW)

   axes[0,1].text(0, wp_tl[0]+.05, f"{wp_tl[0]:.2f}", ha="left", va="bottom")
   axes[0,1].text(len(wp_tl), wp_tl[-1]+.05, f"{wp_tl[-1]:.2f}", ha="right", va="bottom")

   axes[1,1].text(0, wp_f1[0]-.05, f"{100*wp_f1[0]:.1f}%", ha="left", va="top")
   axes[1,1].text(len(wp_f1), wp_f1[-1]-.05, f"{100*wp_f1[-1]:.1f}%", ha="right", va="top")

   # sgd best possible
   r = Results()
   r.load(loc_sgd)

   best_tl = min(r.train_losses)
   best_f1 = max(r.f1_scores)

   f1_rel_recovery = (f1_after_recovery - f1_after_fault) / (best_f1 - f1_after_fault)

   axes[0,1].axhline(best_tl, linestyle="--", color=YELLOW, linewidth=LW, label="SGD")
   axes[1,1].axhline(best_f1, linestyle="--", color=YELLOW, linewidth=LW)

   # tidy up
   axes[0,0].set_xticks(())
   axes[0,1].set_xticks(())
   axes[0,1].set_yticks(())
   axes[1,1].set_yticks(())
   axes[1,0].set_xlabel("Epochs")
   axes[1,1].set_xlabel("Perturbation steps")
   axes[0,0].set_ylabel("Train loss")
   axes[1,0].set_ylabel("$F_1$ score")
   axes[0,0].set_title("Baseline")
   axes[0,1].set_title("Recovery from fault")

   min_y_tl = min(min(tl), min(wp_tl), best_tl)
   max_y_tl = max(max(tl), max(wp_tl), best_tl)
   min_y_f1 = min(min(f1), min(wp_f1), best_f1)
   max_y_f1 = max(max(f1), max(wp_f1), best_f1)

   ylim_tl = (min_y_tl-.1, 1.05*max_y_tl)
   ylim_f1 = (0, 1.05)

   axes[0,0].set_ylim(ylim_tl)
   axes[0,1].set_ylim(ylim_tl)
   axes[1,0].set_ylim(ylim_f1)
   axes[1,1].set_ylim(ylim_f1)

   axes[0,0].legend()
   axes[0,1].legend()

   title = f"{h}x{w}x{d} 1-to-{c} with {b} branches, trial {trial}, recovery on {dataset.value}"
   plt.suptitle(title)

   # render
   fig.tight_layout()
   fname = f"{h}x{w}x{d}@{c}-{safe_b}.{dataset.name}.t{trial}.png"
   save_fig(plt, f"fig_wp_scores/{PERCENT}percent/{safe_b}/", fname, show)

   # code
   short_caption = (
      f"Recovery from fault by weight perturbation, "
      f"for {h}x{w}x{d} with {b} branches, trial {trial}"
   )
   caption = (
      f"Model recovery from {PERCENT}\\% applied fault by {wp_steps} steps of "
      f"weight perturbation, for a {h}x{w}x{d} BCN with {b} branches (trial {trial}), "
      f"evaluated on {dataset.value}. "
      f"(a) $F_1$ score recovered from {100*f1_after_fault:.1f}\\% to "
      f"{100*f1_after_recovery:.1f}\\%, a change of {f1_rel_recovery:+.2f} "
      f"relative to the reference. "
      f"(b) Connected core (\\textcolor{{bluecomment}}{{blue}}), and locations of dead neurons "
      f"(\\textcolor{{redcomment}}{{red}})."
   )

   lines = [
      "% Generated by ``scripts/fig_wp_scores.py``",
      "\\begin{figure}[h]",
      "\\centering",
      "\\begin{subfigure}[b]{\\textwidth}",
      "\\centering",
      f"\\includegraphics[width=\\textwidth]{{{h}x{w}x{d}@{c}-{safe_b}.{dataset.name}.t{trial}.png}}",
      "\\caption{Recovery curves}",
      f"\\label{{fig:recovery_{h}x{w}x{d}@{c}-{safe_b}.{dataset.name}.t{trial}:scores}}",
      "\\end{subfigure}",
      "\\hfill",
      "\\begin{subfigure}[b]{\\textwidth}",
      "\\centering",
      f"\\includegraphics[width=\\textwidth]{{{h}x{w}x{d}@{c}-{safe_b}.{dataset.name}.t{trial}.fault.png}}",
      "\\caption{Connected core and fault}",
      f"\\label{{fig:recovery_{h}x{w}x{d}@{c}-{safe_b}.{dataset.name}.t{trial}:fault}}",
      "\\end{subfigure}",
      f"\\caption[{short_caption}]{{{caption}}}",
      f"\\label{{fig:recovery_{h}x{w}x{d}@{c}-{safe_b}.{dataset.name}.t{trial}}}",
      "\\end{figure}",
   ]

   code = "\n".join(lines)
   print("```latex")
   print(code)
   print("```")

   return code


if __name__ == "__main__":
   _ = main()