
from pathlib import Path

from bcn import BCN

import torch

from bcn import BCN, TrainingScheme, Fault, Dataset, Connections
from bcn.branches import DirectOnly
from bcn.branches.uniform import NearestNeighborOnly, NextToNNOnly
from bcn.branches.uniform import NearestNeighbor, NextToNN
from bcn.branches.informed import Kappa, IndirectOnly

in_loc = Path("./results/")
wp_loc = Path("./results/wp/")

BRANCHES = DirectOnly()
DATASET = Dataset.MNIST
CONNECTIONS = Connections.ONE_TO_9

BATCH_SIZE = 64

STEPS = 1000
EPOCHS = 10
FAULT = 0.02
EPSILON = 0.01

if __name__ == "__main__":

   print("Webhook URL, if desired:")
   webhook = input("> ")

   for trial in (1, 2, 3):
      for width in (16, 30):
         for depth in (3, 6):

            # prepare model
            model = BCN(
               width, depth,
               connections=CONNECTIONS,
               branches=BRANCHES,
               dropout=0,
               verbose=2,
            )
            # prepare for training
            scheme = TrainingScheme(
               dataset=DATASET,
               width=width,
               padding=1,
               batch_size=BATCH_SIZE,
            )
            model.train(
               scheme=scheme,
               trial=trial,
               from_path=in_loc,
               save_path=wp_loc,
               tag=f"{width}x{width}x{depth}, {BRANCHES.name}"
            )
            model_name = model.default_results_filename.replace("results", "fault", 1)

            fault = Fault(model=model, proportion=FAULT)
            torch.save(fault, wp_loc / model_name)

            model_ = model.clone(clone_results=False)
            model_.train(trial=f"{trial}o", save_path=wp_loc)

            model_.run_epochs(EPOCHS, fault=fault, webhook=webhook)
            model.run_wp(STEPS, fault=fault, webhook=webhook)
