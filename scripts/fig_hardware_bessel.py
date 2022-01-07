
import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

from plotutils import save_fig, HIGH_CONTRAST

ORDERS = range(5)

X = np.linspace(0, 7, 50)

FONT_SIZE = 20
plt.rcParams.update({'font.size': FONT_SIZE})

def main():
   n = len(ORDERS)
   max_heights = []
   ys = []

   for k in ORDERS:
      y = jv(k, X) ** 2
      max_heights.append(np.max(y))
      ys.append(y)

   fig, axes = plt.subplots(
      n,
      gridspec_kw={"height_ratios": max_heights},
      figsize=(8,9),
   )
   for k, y, h, ax in zip(ORDERS, ys, max_heights, axes.flatten()):
      ax.plot(X, y, color=HIGH_CONTRAST[1], linewidth=3)

      ax.set_ylim((0, h*1.05))
      ax.set_xlim((X[0], X[-1]))

      ax.set_ylabel(f"$J_{{{k}}}^2$        ", rotation=0)

      if k != ORDERS[-1]:
         ax.set_xticks(())
      else:
         ax.set_xlabel("$\\kappa$")

   plt.tight_layout()
   save_fig(plt, "fig_hardware_bessel/", "bessel.png", True)
   

if __name__ == "__main__":
   _ = main()