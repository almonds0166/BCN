"""
Edge effects gets complicated very quickly
"""

import sys
sys.path.append("../")

import numpy as np

# (h, w, c, n)
SIZES = [
   (16, 16, 9, 1),
   (16, 16, 9, 9),
   (16, 16, 9, 25),
   (16, 16, 25, 1),
   (16, 16, 25, 9),
   (16, 16, 25, 25),
   (30, 30, 9, 1),
   (30, 30, 9, 9),
   (30, 30, 9, 25),
   (30, 30, 25, 1),
   (30, 30, 25, 9),
   (30, 30, 25, 25),
]


MEASURES = ["Neurons", "Weights", "Connections"]

def inbounds(r, c, h, w):
   return 0 <= r < h and 0 <= c < w

def format_shape(shape):
   h, w, c, n = shape
   b = "DIR" if n == 1 else f"N{n}"
   return f"{h}x{w} & 1-to-{c} & {b}"

def main():
   counts = []

   for h, w, c, n in SIZES:
      r_direct = round(np.sqrt(c)) // 2
      r_indirect = round(np.sqrt(n)) // 2
      num_neurons = h*w
      num_weights = 0
      num_connections = 0

      for r in range(h):
         for c in range(w):
            for dr in range(-r_direct, r_direct+1):
               for dc in range(-r_direct, r_direct+1):
                  if not inbounds(r+dr, c+dc, h, w): continue
                  num_weights += 1
                  for ddr in range(-r_indirect, r_indirect+1):
                     for ddc in range(-r_indirect, r_indirect+1):
                        if not inbounds(r+dr+ddr, c+dc+ddc, h, w): continue
                        num_connections += 1

      counts.append((num_neurons, num_weights, num_connections))

   #print(counts)

   lines = [
      "\\begin{table}",
      "\\centering",
      "\\begin{tabular}{ c c c  c c c }",
      "Size & Directed & Branches & " + " & ".join(MEASURES) + "\\\\",
      "\\hline",
   ]

   for shape, (n, w, c) in zip(SIZES, counts):
      line = f"{format_shape(shape)} & {n} & {w} & {c} \\\\"
      lines.append(line)

   lines.extend([
      "\\end{tabular}",
      "\\caption[]{}",
      "\\label{tab:counts}",
      "\\end{table}",
   ])

   code = "\n".join(lines)

   print("```latex")
   print(code)
   print("```")

   return code

if __name__ == "__main__":
   _ = main()