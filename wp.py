
from bcn import BCN

import torch

from bcn import BCN, TrainingScheme, Fault, Dataset, Connections
from bcn.branches import DirectOnly
from bcn.branches.uniform import NearestNeighborOnly, NextToNNOnly
from bcn.branches.uniform import NearestNeighbor, NextToNN
from bcn.branches.informed import Kappa, IndirectOnly

in_loc = "./results/"
wp_loc = "./results/wp/"

BRANCHES = DirectOnly()
DATASET = Dataset.MNIST
CONNECTIONS = Connections.ONE_TO_9

BATCH_SIZE = 256

STEPS = 1000
FAULT = 0.005
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
   
            fault = Fault(model, proportion=FAULT)
            model.run_wp(STEPS, fault=fault, webhook=webhook)
