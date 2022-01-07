

import sys
sys.path.append("../")

from pathlib import Path

import torch
from bcn import BCN, Results, Dataset, Connections, Fault
from bcn.branches import DirectOnly
from bcn.branches.uniform import NearestNeighbor, NearestNeighborOnly, NextToNN, NextToNNOnly
from bcn.branches.informed import Kappa, IndirectOnly

from plotutils import (
   plot_fault,
   ABBREVIATIONS, ABBREVIATIONS_SAFE, save_fig,
   Cmaps,
)

ALL_BRANCHES = {
   "DirectOnly": DirectOnly(),
   "uniform.NearestNeighbor": NearestNeighbor(),
   "uniform.NearestNeighborOnly": NearestNeighborOnly(),
   "uniform.NextToNN": NextToNN(),
   "uniform.NextToNNOnly": NextToNNOnly(),
   "informed.Kappa(1.0)": Kappa(1.0),
   "informed.Kappa(1.5)": Kappa(1.5),
   "informed.IndirectOnly": IndirectOnly(),
}

BATCH_SIZE = 64
PERCENT = 2

DATASET = (Dataset.MNIST,)
CONNECTIONS = (Connections.ONE_TO_9,)
BRANCHES = ("DirectOnly",)
HEIGHT = (30,)
DEPTH = (3,)
TRIAL = (3,)

WP_PATH = Path(input("Enter the location of all your *WP* results\n> "))
WP_PATH /= f"{PERCENT}percent" # get subfolder with only the desired results

def render_fault_plot(height, depth, dataset, connections, branches, trial, *, show=False):
   h = height; w = h; d = depth
   c = connections.value
   safe_b = ABBREVIATIONS_SAFE[branches]
   fname = f"fault_{h}x{w}x{d}@{c}-{branches}.{dataset.name}.b{BATCH_SIZE}.t{trial}.pkl"

   fault = torch.load(WP_PATH / fname)
   model = BCN(w, d, connections=connections, branches=ALL_BRANCHES[branches])
   
   plt = plot_fault(fault, model=model)

   fname = f"{h}x{w}x{d}@{c}-{safe_b}.{dataset.name}.t{trial}.fault.png"
   save_fig(plt, f"fig_wp_fault/{PERCENT}percent/{safe_b}/", fname, show)

def main():
   for height in HEIGHT:
      for depth in DEPTH:
         for dataset in DATASET:
            for connections in CONNECTIONS:
               for branches in BRANCHES:
                  for trial in TRIAL:
                     render_fault_plot(height, depth, dataset, connections, branches, trial)

if __name__ == "__main__":
   _=main()