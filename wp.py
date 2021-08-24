
from bcn import BCN

import torch

from bcn import BCN, TrainingScheme, Fault, Dataset, Connections
from bcn.branches import DirectOnly
from bcn.branches.uniform import NearestNeighborOnly, NextToNNOnly
from bcn.branches.uniform import NearestNeighbor, NextToNN
from bcn.branches.informed import Kappa, IndirectOnly

from plotutils import plot_fault

in_loc = "./results/"
wp_loc = "./results/wp/"

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

   for trial in (1,):
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
            model_name = repr(model).replace("<", "_").replace(">", "")
            fig_file = "./results/wp/" + model_name + ".png"

            fault = Fault(model=model, proportion=FAULT)
            plot_fault(fault, save_file=fig_file)

            model_ = model.clone(clone_results=False)
            model_.train(trial=f"{trial}o", save_path=wp_loc)

            model_.run_epochs(EPOCHS, fault=fault, webhook=webhook)
            model.run_wp(STEPS, fault=fault, webhook=webhook)
