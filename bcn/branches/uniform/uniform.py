
from ..branches import Branches, DirectOnly

class NearestNeighbor(Branches):
   """Branches class representing nearest neighbor connections.

   For nearest neighbor connections, each arm has 9 fingers.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

      o = self.center
      
      self.default[o-1:o+2,o-1:o+2] = 1

      self.normalize()

   def __repr__(self):
      return f"{self.module}.{self.__class__.__name__}()"

class NearestNeighbour(NearestNeighbor):
   """Alias for ``NearestNeighbor``.
   """
   pass

class NextToNN(Branches):
   """Branches class representing next-to-nearest neighbor connections.

   For next-to-nearest neighbor connections, each arm has 25 fingers.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

      o = self.center

      self.default[o-2:o+3,o-2:o+3] = 1

      self.normalize()

   def __repr__(self):
      return f"{self.module}.{self.__class__.__name__}()"

class NearestNeighborOnly(Branches):
   """Branches class representing nearest neighbor connections without the center connection.

   For this connection scheme, each arm has 8 fingers touching the corresponding first ring of
   indirect target neurons.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

      o = self.center

      self.default[o-1:o+2,o-1:o+2] = 1
      self.default[o,o] = 0

      self.normalize()

   def __repr__(self):
      return f"{self.module}.{self.__class__.__name__}()"

class NearestNeighbourOnly(NearestNeighborOnly):
   """Alias for ``NearestNeighborOnly``.
   """
   pass

class NextToNNOnly(Branches):
   """Branches class representing next-to-nearest neighbor connections without the innermost rings.

   For this connection scheme, each arm has 16 fingers touching the corresponding second ring of
   indirect target neurons.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

      o = self.center

      self.default[o-2:o+3,o-2:o+3] = 1
      self.default[o-1:o+2,o-1:o+2] = 0

      self.normalize()

   def __repr__(self):
      return f"{self.module}.{self.__class__.__name__}()"

class IndirectOnly(Branches):
   """Nearest and next-to-nearest neighbor Branches class, without the center connection.

   For this connection scheme, each arm has 24 fingers, touching the corresponding first two rings
   of indirect target neurons.
   """
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

      o = self.center

      self.default[o-2:o+3,o-2:o+3] = 1
      self.default[o,o] = 0
      
      self.normalize()

   def __repr__(self):
      return f"{self.module}.{self.__class__.__name__}()"