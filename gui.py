
import sys

from PyQt5 import QtCore, QtGui, QtWidgets #, uic
import numpy as np

from ui.main import Ui_MainWindow
import bcn
from bcn.branches import Branches, DirectOnly, Vital
from bcn.branches import simple, uniform

TABLE_WIDGETS = lambda ui: {
   (-2,-2): ui.tw_PP,
   (-2,-1): ui.tw_PT,
   (-2, 0): ui.tw_P0,
   (-2, 1): ui.tw_P1,
   (-2, 2): ui.tw_P2,
   (-1,-2): ui.tw_TP,
   (-1,-1): ui.tw_TT,
   (-1, 0): ui.tw_T0,
   (-1, 1): ui.tw_T1,
   (-1, 2): ui.tw_T2,
   ( 0,-2): ui.tw_0P,
   ( 0,-1): ui.tw_0T,
   ( 0, 0): ui.tw_00,
   ( 0, 1): ui.tw_01,
   ( 0, 2): ui.tw_02,
   ( 1,-2): ui.tw_1P,
   ( 1,-1): ui.tw_1T,
   ( 1, 0): ui.tw_10,
   ( 1, 1): ui.tw_11,
   ( 1, 2): ui.tw_12,
   ( 2,-2): ui.tw_2P,
   ( 2,-1): ui.tw_2T,
   ( 2, 0): ui.tw_20,
   ( 2, 1): ui.tw_21,
   ( 2, 2): ui.tw_22,
}

class Handles:
   def click_load():
      """Handles when the user presses the Load button under Branch matrices tools"""
      selected = ui.cb_branches.currentText()

      selected_to_branches = {
         "DirectOnly": DirectOnly,
         "Vital": Vital,
         "uniform.NearestNeighbor": uniform.NearestNeighbor,
         "uniform.NearestNeighborOnly": uniform.NearestNeighborOnly,
         "uniform.NextToNN": uniform.NextToNN,
         "uniform.NextToNNOnly": uniform.NextToNNOnly,
         "uniform.IndirectOnly": uniform.IndirectOnly,
         "simple.NearestNeighbor": simple.NearestNeighbor,
         "simple.NearestNeighborOnly": simple.NearestNeighborOnly,
         "simple.NextToNN": simple.NextToNN,
         "simple.NextToNNOnly": simple.NextToNNOnly,
         "simple.IndirectOnly": simple.IndirectOnly,
      }

      branches = selected_to_branches[selected]()

      table_widgets = TABLE_WIDGETS(ui)

      o = branches.center
      for dx in range(-2,2+1):
         for dy in range(-2,2+1):
            for i in range(-2,2+1):
               for j in range(-2,2+1):
                  value = branches[dy,dx][o+i,o+j].cpu().numpy()
                  item = table_widgets[dy,dx].item(2+i, 2+j)
                  if value == 0:
                     item.setText(QtCore.QCoreApplication.translate("MainWindow", ""))
                  else:
                     item.setText(QtCore.QCoreApplication.translate(
                        "MainWindow",
                        str(value).rstrip('0').rstrip('.')
                     ))

   def click_compute():
      """Handles the Compute button."""
      allow_user_input(ui, False)
      W = get_weights(ui)
      B = get_branches(ui)
      c = get_connections(ui)
      A = bcn.BCN.construct_network(12, c, B)
      I = get_input(ui).reshape(144,1)
      # forward
      O = np.zeros(I.shape)
      for dy,dx in A.keys():
         o = A[dy,dx] @ (I * W[2+dy,2+dx])
         O = np.add(O, o)
      O = O.reshape(12,12)
      set_output(ui, O)
      allow_user_input(ui, True)

def allow_user_input(ui, allow: bool) -> None:
   """Enable or disable user input"""
   for x in (
      #ui.checkBox,
      #ui.btn_norm,
      ui.btn_load,
      ui.tw_weights,
      ui.cb_direct,
      ui.tw_PP, ui.tw_PT, ui.tw_P0, ui.tw_P1, ui.tw_P2,
      ui.tw_TP, ui.tw_TT, ui.tw_T0, ui.tw_T1, ui.tw_T2,
      ui.tw_0P, ui.tw_0T, ui.tw_00, ui.tw_01, ui.tw_02,
      ui.tw_1P, ui.tw_1T, ui.tw_10, ui.tw_11, ui.tw_12,
      ui.tw_2P, ui.tw_2T, ui.tw_20, ui.tw_21, ui.tw_22,
      ui.tw_input,
      ui.tw_output,
   ):
      x.setEnabled(allow)

def get_weights(ui) -> np.array:
   """Get the weights as a numpy matrix."""
   W = np.zeros((5,5))
   for i in range(5):
      for j in range(5):
         item = ui.tw_weights.item(i, j).text().strip()
         if item:
            try:
               W[i,j] = float(item)
            except ValueError:
               pass

   return W

def get_branches(ui) -> Branches:
   """Get the branch connection values as a Branches object."""
   B = Branches(5)
   table_widgets = TABLE_WIDGETS(ui)
   for dx in range(-2,2+1):
      for dy in range(-2,2+1):
         tw = table_widgets[dy,dx]
         B[dy,dx] = np.zeros((5,5))
         for i in range(5):
            for j in range(5):
               item = tw.item(i, j).text().strip()
               if item:
                  try:
                     B[dy,dx][i,j] = float(item)
                  except ValueError:
                     pass

   return B

def get_connections(ui) -> bcn.Connections:
   """Get the selected type of connections."""
   selected = ui.cb_direct.currentText()
   if selected == "ONE_TO_9": return bcn.Connections.ONE_TO_9
   if selected == "ONE_TO_25": return bcn.Connections.ONE_TO_25

def get_input(ui) -> np.array:
   """Get the input matrix"""
   I = np.zeros((12,12))
   for i in range(12):
      for j in range(12):
         item = ui.tw_input.item(i, j).text().strip()
         if item:
            try:
               I[i,j] = float(item)
            except ValueError:
               pass

   return I

def set_output(ui, O) -> None:
   """Given matrix O, write to the output table widget"""
   for i in range(12):
      for j in range(12):
         value = O[i,j].cpu().numpy()
         item = ui.tw_output.item(i, j)
         if value == 0:
            item.setText(QtCore.QCoreApplication.translate("MainWindow", ""))
         else:
            item.setText(QtCore.QCoreApplication.translate(
               "MainWindow",
               str(value).rstrip('0').rstrip('.')
            ))

if __name__ == "__main__":
   app = QtWidgets.QApplication(sys.argv)
   MainWindow = QtWidgets.QMainWindow()
   ui = Ui_MainWindow()
   ui.setupUi(MainWindow)

   ui.actionQuit.triggered.connect(MainWindow.close)
   ui.btn_load.clicked.connect(Handles.click_load)
   ui.btn_in2out.clicked.connect(Handles.click_compute)

   MainWindow.show()
   sys.exit(app.exec_())
