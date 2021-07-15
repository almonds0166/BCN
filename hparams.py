
import torch

from bcn import BCN, TrainingScheme
from bcn import Dataset, Connections
from bcn.branches.simple import DirectOnly, NearestNeighborOnly, NextToNNOnly

BRANCHES = DirectOnly()
DATASET = Dataset.MNIST
CONNECTIONS = Connections.ONE_TO_9

BATCH_SIZE = 64
NUM_EPOCHS = 50 # recommend >=30 for DirectOnly, >= 60 for NextToNN
OPTIMIZER = torch.optim.Adam

if __name__ == "__main__":

   print("Webhook URL, if desired:")
   webhook = input("> ")

   for width in (30, 16):
      for depth in (6, 3):
         for trial in (1, 2, 3):
            # prepare model
            model = BCN(
               width, depth,
               connections=CONNECTIONS,
               branches=BRANCHES,
               dropout=0,
               verbose=True,
            )
            # prepare for training
            scheme = TrainingScheme(
               dataset=DATASET,
               width=width,
               padding=1,
               batch_size=BATCH_SIZE,
               optim=OPTIMIZER
            )
            model.train(
               scheme=scheme,
               trial=trial,
               save_path="./results/",
               tag=f"{width}x{width}x{depth}, {BRANCHES.name}"
            )
            
            model.run_epochs(NUM_EPOCHS, webhook=webhook)

   print("Done.")