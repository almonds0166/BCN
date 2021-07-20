
from scipy.special import jv

from ..branches import Branches, DirectOnly

class Kappa(Branches):
   """Grating strength
   """
   def __init__(self, kappa: float):
      super().__init__()

      self.kappa = float(kappa) # convert any ints
      
      rings = {
         0:   jv(0, kappa)**2,
         1: 2*jv(1, kappa)**2 / 8,
         2: 2*jv(2, kappa)**2 / 16,
      }
      o = self.center
      for d in sorted(rings, reverse=True):
         self.default[o-d:o+d+1,o-d:o+d+1] = rings[d]

      self.normalize()

   def __repr__(self):
      return f"{self.module}.{self.__class__.__name__}({self.kappa})"

   def __str__(self):
      return repr(self)

   @property
   def name(self):
      return f"Grating strength {self.kappa:.02f}"

class IndirectOnly(Kappa):
   """Construct a Kappa object where the direct term is 0
   """
   def __init__(self):
      super().__init__(2.4048)

   def __repr__(self):
      return str(self) + "()"

   def __str__(self):
      return f"{self.module}.{self.__class__.__name__}"

   @property
   def name(self):
      return "Grating strength 2.4048"
