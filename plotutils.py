
import numpy as np
from matplotlib import pyplot as plt

# figure settings
FONT_SIZE = 14

# From Paul Tol's colour schemes, circa April 2021, https://personal.sron.nl/~pault/
HIGH_CONTRAST = (
   "#004488", # blue
   "#BB5566", # red
   "#DDAA33", # yellow
)

def plot_loss(results, title: str, height_inches: float=10, width_inches: float=10):
   """Plot model loss results.

   Args:
      results: Either a bcn.Results object or a matrix of bcn.Reults objects.
      title: Overal figure title.
      height_inches: Height of figure, in inches.
      width_inches: Width of figure, in inches.

   Returns:
      plt.
   """
   plt.rcParams.update({'font.size': FONT_SIZE})

   R = np.array(results) # does nothing if already a np array
   assert R.ndim in (1, 2, 3), \
      f"results matrix should have dimension of no more than 3, yet results.ndim = {R.ndim}."
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
         ax = axes[i,j]
         ax.errorbar(
            range(epochs),
            train_avg[i,j],
            yerr=[train_min[i,j], train_max[i,j]],
            color=HIGH_CONTRAST[0],
            label="Train",
            linewidth=3
         )
         ax.errorbar(
            range(epochs),
            valid_avg[i,j],
            yerr=[valid_min[i,j], valid_max[i,j]],
            color=HIGH_CONTRAST[1],
            label="Valid",
            linewidth=3
         )
         ax.set_ylim((-0.05, 2.80))
         ax.set_title(R[i,j,0].tag)
         if j == 0: ax.set_ylabel("Loss")
         if i == h-1: ax.set_xlabel("Epoch")
         if (i,j) == (h-1,w-1): ax.legend()

   fig.suptitle(title)
   fig.tight_layout()

   return plt.show()

def plot_f1_scores(results, title: str, height_inches: float=10, width_inches: float=10):
   """Plot model F1 scores of results.

   Args:
      results: Either a bcn.Results object or a matrix of bcn.Reults objects.
      title: Overal figure title.
      height_inches: Height of figure, in inches.
      width_inches: Width of figure, in inches.

   Returns:
      plt.
   """
   plt.rcParams.update({'font.size': FONT_SIZE})

   R = np.array(results) # does nothing if already a np array
   assert R.ndim in (1, 2, 3), \
      f"results matrix should have dimension of no more than 3, yet results.ndim = {R.ndim}."
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
   for i in range(h):
      for j in range(w):
         for e in range(epochs):
            for t in range(trials):
               f1_min[i,j,e] = min(f1_min[i,j,e], R[i,j,t].f1_scores[e])
               f1_max[i,j,e] = max(f1_max[i,j,e], R[i,j,t].f1_scores[e])
               f1_avg[i,j,e] += R[i,j,t].f1_scores[e]
            f1_avg[i,j,e] /= trials
            f1_min[i,j,e] = f1_avg[i,j,e] - f1_min[i,j,e]
            f1_max[i,j,e] = f1_max[i,j,e] - f1_avg[i,j,e]

   fig, axes = plt.subplots(h,w)
   fig.set_size_inches(width_inches, height_inches)

   for i in range(h):
      for j in range(w):
         ax = axes[i,j]
         ax.errorbar(
            range(epochs),
            f1_avg[i,j],
            yerr=[f1_min[i,j], f1_max[i,j]],
            color=HIGH_CONTRAST[2],
            label="F1 score",
            linewidth=3
         )
         ax.set_ylim((-0.05, 1.05))
         ax.set_title(R[i,j,0].tag)
         if j == 0: ax.set_ylabel("F1 score")
         if i == h-1: ax.set_xlabel("Epoch")
         #if (i,j) == (h-1,w-1): ax.legend()

   fig.suptitle(title)
   fig.tight_layout()

   return plt.show()



if __name__ == "__main__":
   pass