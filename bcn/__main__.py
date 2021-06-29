
import sys

import bcn

if __name__ == "__main__":

   print("Python v{0.major}.{0.minor}.{0.micro}".format(sys.version_info), end=", ")
   print("BCN v{0.major}.{0.minor}.{0.build}".format(bcn.version_info))
   print("So far, so good.")

   print("Trying out a quick toy test:")
   print("\tTraining a 10x10x1@9-DirectOnly model on MNIST for 3 epochs.")
   arg = sys.argv[1].lower()
   if arg == "save":
      print("\tAnd saving them to ./results/")
   elif arg == "load":
      print("\tAnd continuing off from ./results/")

   model = bcn.BCN(10, 1, verbose=True)
   scheme = bcn.TrainingScheme(width=10)
   model.train(scheme=scheme)
   if arg == "save":
      model.train(save_path="./results/")
   elif arg == "load":
      model.train(from_path="./results/")
   model.run_epochs(3)

   print("Results:")
   for k, v in model.results:
      print(f"\t{k}: {v}")