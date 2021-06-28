
import re
from pathlib import Path
from math import ceil

from bcn import Results, Dataset
from bcn.branches import BRANCH_NAMES

class Info:
   REX = re.compile(
      r"results_" \
      r"([0-9]+)x([0-9]+)x([0-9]+)" \
      r"@([0-9]+)-(\w+)\." \
      r"(\w+)\.b([0-9]+)\.t(\w+)\.pkl",
      re.IGNORECASE
   )

   def __init__(self, fname):
      self.fname = fname
      self.height = None
      self.width = None
      self.depth = None
      self.connections = None
      self.branches = None
      self.dataset = None
      self.batch_size = None
      self.trial = None
      for match in self.REX.finditer(fname):
         self.height = match.group(1)
         self.width = match.group(2)
         self.depth = match.group(3)
         self.connections = match.group(4)
         self.branches = match.group(5)
         self.dataset = match.group(6)
         self.batch_size = match.group(7)
         self.trial = match.group(8)

      for k, v in self.__dict__.items():
         if isinstance(v, str):
            try:
               self.__dict__[k] = int(v)
            except:
               try:
                  self.__dict__[k] = float(v)
               except:
                  pass

   def __repr__(self):
      return f"{self.__class__.__name__}({self.fname})"

LOCATION = "./results/"
CONNECTIONS = 9
DATASET = "MNIST"
DATASET = Dataset(DATASET)

if __name__ == "__main__":

   print(f"Results for {DATASET.value} at 1-to-{CONNECTIONS} connections.\n")
   label = f"Results{DATASET.name}at{CONNECTIONS}"

   data = {}
   for file in Path(LOCATION).iterdir():
      fname = file.name
      if fname[-4:] != ".pkl": continue

      info = Info(fname)
      if info.connections != CONNECTIONS: continue
      if info.dataset != DATASET.name: continue

      branches = BRANCH_NAMES[info.branches]
      entry = (
         f"{info.height}x{info.width}x{info.depth} ({branches})"
      )

      if entry not in data:
         data[entry] = {
            "f1": 0,
            "vl": float("inf"),
         }

      r = Results()
      r.load(file)

      data[entry]["f1"] = max(data[entry]["f1"], r.f1_scores[r.best_epoch])
      data[entry]["vl"] = min(data[entry]["vl"], r.valid_losses[r.best_epoch])

   print("\\begin{{center}}")
   print("\\begin{{tabular}}{{ c c c }}")
   print("\\hline")
   print("Model & Minimum encountered validation loss & F1 score \\\\")
   print("\\hline")
   for entry in sorted(list(data.keys())):
      print("{entry} & {vl:.3f} & {f1:.3f} \\\\".format(
         entry=entry,
         **data[entry]
      ))
   print("\\hline")
   print("\\end{{tabular}}")
   print("\\caption{{Insert caption!}}")
   print("\\label{{table:{}}}".format(label))
   print("\\end{{center}}")




