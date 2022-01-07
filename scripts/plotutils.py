
from pathlib import Path

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from bcn import BCN
import bcn.branches

# figure settings
FONT_SIZE = 14

# From Paul Tol's colour schemes, circa April 2021, https://personal.sron.nl/~pault/
HIGH_CONTRAST = (
   "#004488", # blue
   "#BB5566", # red
   "#DDAA33", # yellow
)

MNIST_COLOR = "#BDD9BF"
FASHION_COLOR = "#A997DF"

BAR_EPS = 0.15
ALPHA = 0.2

BRANCH_NAMES = {
   "DirectOnly": "Direct connections only",
   "informed.Kappa(1.0)": "Intermediate branches configuration a",
   "informed.Kappa(1.5)": "Intermediate branches configuration b",
   "informed.IndirectOnly": "Extinguished direct connections",
   "uniform.NearestNeighborOnly": "First ring only",
   "uniform.NextToNNOnly": "Second ring only",
   "uniform.NearestNeighbor": "Nearest neighbors connections",
   "uniform.NextToNN": "Next-to-nearest neighbors connections",
}

ABBREVIATIONS = {
   "DirectOnly": "Dir",
   "informed.Kappa(1.0)": "BCa",
   "informed.Kappa(1.5)": "BCb",
   "informed.IndirectOnly": "Ext",
   "uniform.NearestNeighborOnly": "FRO",
   "uniform.NextToNNOnly": "SRO",
   "uniform.NearestNeighbor": "N9",
   "uniform.NextToNN": "N25",
}

# deprecated
ABBREVIATIONS_SAFE = {
   "DirectOnly": "DIR",
   "informed.Kappa(1.0)": "k(1.0)",
   "informed.Kappa(1.5)": "k(1.5)",
   "informed.IndirectOnly": "k(2.4)",
   "uniform.NearestNeighborOnly": "FRO",
   "uniform.NextToNNOnly": "SRO",
   "uniform.NearestNeighbor": "N9",
   "uniform.NextToNN": "N25",
}

BRANCHES = {
   "DirectOnly": bcn.branches.DirectOnly(),
   "informed.Kappa(1.0)": bcn.branches.informed.Kappa(1.0),
   "informed.Kappa(1.5)": bcn.branches.informed.Kappa(1.5),
   "informed.IndirectOnly": bcn.branches.informed.IndirectOnly(),
   "uniform.NearestNeighborOnly": bcn.branches.uniform.NearestNeighborOnly(),
   "uniform.NextToNNOnly": bcn.branches.uniform.NextToNNOnly(),
   "uniform.NearestNeighbor": bcn.branches.uniform.NearestNeighbor(),
   "uniform.NextToNN": bcn.branches.uniform.NextToNN(),
}

class Colors:
   BLUE = tuple(i/255 for i in (68,119,170))
   RED = tuple(i/255 for i in (204,102,119))
   YELLOW = tuple(i/255 for i in (221,204,119))
   GRAY = tuple(i/255 for i in (130,130,130))

def cmap_from_color(positive_color, negative_color=None,
   *, N=100, name="mycmap"):
   """Linear colormap from given color

   Args:
      color (tuple[float,float,float]): The color as a tuple of floats in ``0..1``

   Keyword args:
      N (int): The number of colormap quantization levels, default 10.
      name (str): The (internal) name of the cmap. Default "mycmap".
   """
   positive = [(*positive_color, c) for c in np.linspace(0,1,N)]
   if negative_color is None:
      return mcolors.LinearSegmentedColormap.from_list(name, positive, N=N)
   
   negative = [(*negative_color, c) for c in np.linspace(1,0,N)]
   _ = negative + positive
   return mcolors.LinearSegmentedColormap.from_list(name, _, N=N)

class Cmaps:
   BLUE = cmap_from_color(Colors.BLUE, name="myblue")
   RED = cmap_from_color(Colors.RED, name="myred")
   YELLOW = cmap_from_color(Colors.YELLOW, name="myyellow")
   GRAY = cmap_from_color(Colors.GRAY, name="mygray")
   BLUE_TO_RED = cmap_from_color(Colors.RED, Colors.BLUE, name="myBuRd")
   RED_TO_BLUE = cmap_from_color(Colors.BLUE, Colors.RED, name="myRdBu")

def save_fig(plt, folder, filename, show=False):
   """Given a matplotlib plot and location, save the figure
   """
   output_folder = Path(folder)
   output_folder.mkdir(parents=True, exist_ok=True)
   plt.savefig(output_folder / filename)
   
   if show: plt.show()

def plot_loss(results, title: str,
   height_inches: float=10, width_inches: float=10,
   titles=None,
):
   """Plot model loss results.

   Args:
      results: Either a bcn.Results object or a matrix of bcn.Reults objects.
      title: Overal figure title.
      height_inches: Height of figure, in inches.
      width_inches: Width of figure, in inches.
      titles: List of subplot titles to replace the results tags. An element of None will use the
         results tag if available. Must be in the same shape as the subplots.

   Returns:
      plt.
   """
   plt.rcParams.update({'font.size': FONT_SIZE})

   R = np.array(results) # does nothing if already a np array
   assert R.ndim in (1, 2, 3), \
      f"results matrix should have dimension of no more than 3, yet {R.ndim=}."
   if R.ndim == 1:
      R = np.expand_dims(R, 0)
   if R.ndim == 2:
      R = np.expand_dims(R, -1)

   h, w, trials = R.shape

   epochs = len(R.flat[0].train_losses)

   # get min and max losses in an admittedly lame way, so long as it works
   train_min = np.tile(float("inf"),  (h,w,epochs))
   train_max = np.tile(float("-inf"), (h,w,epochs))
   train_avg = np.zeros((h,w,epochs))
   valid_min = np.tile(float("inf"),  (h,w,epochs))
   valid_max = np.tile(float("-inf"), (h,w,epochs))
   valid_avg = np.zeros((h,w,epochs))
   for i in range(h):
      for j in range(w):
         for e in range(epochs):
            for t in range(trials):
               train_min[i,j,e] = min(train_min[i,j,e], R[i,j,t].train_losses[e])
               train_max[i,j,e] = max(train_max[i,j,e], R[i,j,t].train_losses[e])
               train_avg[i,j,e] += R[i,j,t].train_losses[e]
               valid_min[i,j,e] = min(valid_min[i,j,e], R[i,j,t].valid_losses[e])
               valid_max[i,j,e] = max(valid_max[i,j,e], R[i,j,t].valid_losses[e])
               valid_avg[i,j,e] += R[i,j,t].valid_losses[e]
            train_avg[i,j,e] /= trials
            train_min[i,j,e] = train_avg[i,j,e] - train_min[i,j,e]
            train_max[i,j,e] = train_max[i,j,e] - train_avg[i,j,e]
            valid_avg[i,j,e] /= trials
            valid_min[i,j,e] = valid_avg[i,j,e] - valid_min[i,j,e]
            valid_max[i,j,e] = valid_max[i,j,e] - valid_avg[i,j,e]

   fig, axes = plt.subplots(h,w)
   fig.set_size_inches(width_inches, height_inches)

   for i in range(h):
      for j in range(w):
         if w == 1:
            ax = axes[i]
         else:
            ax = axes[i,j]

         x = np.arange(-BAR_EPS, epochs-BAR_EPS)
         ax.fill_between(
            x,
            train_avg[i,j]-train_min[i,j], train_avg[i,j]+train_max[i,j],
            alpha=ALPHA,
            color=HIGH_CONTRAST[0]
         )
         ax.fill_between(
            x,
            valid_avg[i,j]-valid_min[i,j], valid_avg[i,j]+valid_max[i,j],
            alpha=ALPHA,
            color=HIGH_CONTRAST[1]
         )
         ax.plot(
            x,
            train_avg[i,j],
            color=HIGH_CONTRAST[0],
            label="Train",
            linewidth=1,
         )
         ax.errorbar(
            x,
            valid_avg[i,j],
            color=HIGH_CONTRAST[1],
            label="Valid",
            linewidth=1,
         )
         ax.set_ylim((-0.05, 2.80))
         if titles is not None and titles[i,j] is not None:
            ax.set_title(titles[i,j])
         else:
            ax.set_title(R[i,j,0].tag)
         ax.grid(color="lightgray", ls="--")
         # text annotations
         full = len(valid_avg[i,j])-1
         half = full // 2

         s = f"{valid_avg[i,j][half]:.3f}"
         ax.text(
            half, valid_avg[i,j][half]+valid_max[i,j][half]+0.15, s,
            ha="center",
            va="center",
            color="gray",
         )
         s = f"{valid_avg[i,j][full]:.3f}"
         ax.text(
            full, valid_avg[i,j][full]+valid_max[i,j][full]+0.15, s,
            ha="center",
            va="center",
         )
         # labels
         if j == 0: ax.set_ylabel("Loss")
         if i == h-1: ax.set_xlabel("Epoch")
         if (i,j) == (h-1,w-1): ax.legend(loc="upper right")

   fig.suptitle(title)
   #fig.tight_layout()

   return plt.show()

def plot_f1_scores(results, title: str,
   height_inches: float=10, width_inches: float=10,
   titles=None,
):
   """Plot model F1 scores of results.

   Args:
      results: Either a bcn.Results object or a matrix of bcn.Reults objects.
      title: Overal figure title.
      height_inches: Height of figure, in inches.
      width_inches: Width of figure, in inches.
      titles: List of subplot titles to replace the results tags. An element of None will use the
         results tag if available. Must be in the same shape as the subplots.

   Returns:
      plt.
   """
   plt.rcParams.update({'font.size': FONT_SIZE})

   R = np.array(results) # does nothing if already a np array
   assert R.ndim in (1, 2, 3), \
      f"results matrix should have dimension of no more than 3, yet {R.ndim=}."
   if R.ndim == 1:
      R = np.expand_dims(R, 0)
   if R.ndim == 2:
      R = np.expand_dims(R, -1)

   h, w, trials = R.shape

   epochs = len(R.flat[0].train_losses)

   # ...
   f1_min = np.tile(float("inf"),  (h,w,epochs))
   f1_max = np.tile(float("-inf"), (h,w,epochs))
   f1_avg = np.zeros((h,w,epochs))
   ac_min = np.tile(float("inf"),  (h,w,epochs))
   ac_max = np.tile(float("-inf"), (h,w,epochs))
   ac_avg = np.zeros((h,w,epochs))
   for i in range(h):
      for j in range(w):
         for e in range(epochs):
            for t in range(trials):
               f1_min[i,j,e] = min(f1_min[i,j,e], R[i,j,t].f1_scores[e])
               f1_max[i,j,e] = max(f1_max[i,j,e], R[i,j,t].f1_scores[e])
               f1_avg[i,j,e] += R[i,j,t].f1_scores[e]
               ac_min[i,j,e] = min(ac_min[i,j,e], R[i,j,t].accuracies[e])
               ac_max[i,j,e] = max(ac_max[i,j,e], R[i,j,t].accuracies[e])
               ac_avg[i,j,e] += R[i,j,t].accuracies[e]
            f1_avg[i,j,e] /= trials
            ac_avg[i,j,e] /= trials
            f1_min[i,j,e] = f1_avg[i,j,e] - f1_min[i,j,e]
            f1_max[i,j,e] = f1_max[i,j,e] - f1_avg[i,j,e]
            ac_min[i,j,e] = ac_avg[i,j,e] - ac_min[i,j,e]
            ac_max[i,j,e] = ac_max[i,j,e] - ac_avg[i,j,e]

   fig, axes = plt.subplots(h,w)
   fig.set_size_inches(width_inches, height_inches)

   for i in range(h):
      for j in range(w):
         if w == 1:
            ax = axes[i]
         else:
            ax = axes[i,j]

         x = np.arange(1-BAR_EPS, 1+epochs-BAR_EPS)
         ax.fill_between(
            x,
            ac_avg[i,j]-ac_min[i,j], ac_avg[i,j]+ac_max[i,j],
            alpha=ALPHA,
            color=HIGH_CONTRAST[1]
         )
         ax.fill_between(
            x,
            f1_avg[i,j]-f1_min[i,j], f1_avg[i,j]+f1_max[i,j],
            alpha=ALPHA,
            color=HIGH_CONTRAST[0]
         )
         ax.plot(
            x,
            ac_avg[i,j],
            color=HIGH_CONTRAST[1],
            label="Accuracy",
            linewidth=1,
         )
         ax.plot(
            x,
            f1_avg[i,j],
            color=HIGH_CONTRAST[0],
            label="F1 score",
            linewidth=1,
         )
         ax.set_ylim((-0.05, 1.075))
         if titles is not None and titles[i,j] is not None:
            ax.set_title(titles[i,j])
         else:
            ax.set_title(R[i,j,0].tag)
         ax.grid(color="lightgray", ls="--")
         # text annotations
         full = len(f1_avg[i,j])-1
         half = full // 2

         s = f"{100*f1_avg[i,j][half]:.1f}%"
         ax.text(
            half, f1_avg[i,j][half]+f1_max[i,j][half]+0.05, s,
            ha="center",
            va="center",
            color="gray",
         )
         s = f"{100*f1_avg[i,j][full]:.1f}%"
         ax.text(
            full, f1_avg[i,j][full]+f1_max[i,j][full]+0.05, s,
            ha="center",
            va="center",
         )
         # labels
         if j == 0: ax.set_ylabel("Score")
         if i == h-1: ax.set_xlabel("Epoch")
         if (i,j) == (h-1,w-1): ax.legend(loc="lower right")

   fig.suptitle(title)
   #fig.tight_layout()

   return plt.show()

def keypad_connectedness(model):
   """Returns representations of the "connectedness" to the output layer keypad.

   Awful time complexity, as of now.

   Returns a list of numpy arrays representing the model layers where array values represent how
   many output neurons in the keypad layer that corresponding neuron has access to, in 0 to 10
   inclusive.
   """
   # create a model clone that will have unit weights
   w = model.width; hw = w*w
   d = model.depth

   # marks the number of keypad outputs that the neuron is able to reach
   kc = [np.zeros((w,w), dtype=int) for l in range(d)]

   for l in range(d):
      unitary = BCN(
         w, d-l,
         branches=model.branches,
         connections=model.connections,
         activation=torch.relu
      )
      for layer in unitary.layers:
         torch.nn.init.constant_(layer.weights, 1)
      unitary.eval()
      offset = unitary(torch.zeros(hw,1)).detach().numpy()
      for i in range(w):
         for j in range(w):
            #print("\t", i, j)
            in_ = torch.zeros((w,w))
            in_[i,j] = 1
            out = unitary(in_.reshape(hw,1)).detach().numpy() - offset
            kc[l][i,j] = np.count_nonzero(out)

   return kc

def plot_fault(fault, save_file=None, model=None, show=False):
   """Plots a fault mask.

   Args:
      fault: The Fault instance to visualize.
      save_file: Where to save the plot to.
      model: The model to use to color the connectedness of the keypad.
      show: Whether to show the plot or just return it.
   """
   if model:
      kc = keypad_connectedness(model)
   N = len(fault)
   hw = torch.numel(fault[0])
   w = int(np.sqrt(hw))
   fig, axes = plt.subplots(1,N, figsize=(4*N,4))
   for i, mask in enumerate(fault):
      ax = axes[i]
      if model:
         colors = [(0,.53,.74,c) for c in np.linspace(0,1,100)]
         cmapblue = mcolors.LinearSegmentedColormap.from_list("mycmap", colors, N=10)
         im = ax.imshow(kc[i], cmap=cmapblue, vmin=0, vmax=10)
      cmap = mcolors.ListedColormap([(.77,.01,.20,.6), (0,0,0,0)])
      ax.imshow(mask.reshape((w,w)), cmap=cmap)
      ax.set_xticks(tuple())
      ax.set_yticks(tuple())
      ax.set_xticks(np.arange(-.5, w, 1), minor=True)
      ax.set_yticks(np.arange(-.5, w, 1), minor=True)
      ax.grid(which="minor", color="lightgray", linestyle=":")

   if model:
      fig.subplots_adjust(right=0.8)
      cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
      fig.colorbar(im, cax=cbar_ax,) #ticks=range(11))

   if save_file:
      plt.savefig(save_file)
   
   if show: return plt.show()

   return plt

if __name__ == "__main__":
   from bcn import Fault
   fault = Fault(width=30,depth=6,padding=1, proportion=0.01)
   plot_fault(fault)