
import sys

from PyQt5 import QtCore, QtGui, QtWidgets #, uic
import numpy as np

from ui.main import Ui_MainWindow
import bcn
from bcn.branches import Branches, DirectOnly, Vital
from bcn.branches import simple, uniform

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

      for dy in range(-2,2+1):
         for dx in range(-2,2+1):
            KERNELS[dy+2,dx+2,:,:] = branches[dy,dx].cpu().numpy()

      update_kernel(ui)

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

   def select_weight(selected, deselected):
      """Handles when the user selects a different direct connection."""
      global SELECTED_ROW, SELECTED_COL
      index = selected.indexes()[0]

      ui.tw_kernel.setEnabled(True)

      SELECTED_ROW = index.row()
      SELECTED_COL = index.column()

      update_kernel(ui)

   def kernel_update(row, col):
      """Handles when the user edits the kernel."""
      global SELECTED_ROW, SELECTED_COL, KERNELS

      item = ui.tw_kernel.item(row, col)
      if item is None: item = new_item()
      value = item.text().strip()
      if value:
         try:
            KERNELS[SELECTED_ROW,SELECTED_COL,row,col] = float(value)
            item.setText(
               QtCore.QCoreApplication.translate(
                  "MainWindow",
                  str(value).rstrip('0').rstrip('.')
               )
            )
         except ValueError:
            KERNELS[SELECTED_ROW,SELECTED_COL,row,col] = 0
            item.setText(QtCore.QCoreApplication.translate("MainWindow", ""))
         ui.tw_kernel.setItem(row, col, item) # I think this is called even if automated
      else:
         KERNELS[SELECTED_ROW,SELECTED_COL,row,col] = 0

   def click_copypaste():
      """Handles clicking the ^ button, to copy output to input"""
      for i in range(12):
         for j in range(12):
            out_item = ui.tw_output.item(i, j)
            if out_item is None: out_item = new_item()
            in_item = new_item()
            in_item.setText(out_item.text())
            ui.tw_input.setItem(i, j, in_item)

def new_item():
   item = QtWidgets.QTableWidgetItem()
   item.setTextAlignment(QtCore.Qt.AlignCenter)
   return item

def update_kernel(ui):
   """Takes what's in KERNEL and updates tw_kernel"""
   global SELECTED_ROW, SELECTED_COL, KERNELS

   for i in range(9):
      for j in range(9):
         value = KERNELS[SELECTED_ROW,SELECTED_COL,i,j]
         item = ui.tw_kernel.item(i, j)
         if item is None: item = new_item()
         if value == 0:
            item.setText(QtCore.QCoreApplication.translate("MainWindow", ""))
         else:
            item.setText(QtCore.QCoreApplication.translate(
               "MainWindow",
               str(value).rstrip('0').rstrip('.')
            ))
         ui.tw_kernel.setItem(i, j, item)

def allow_user_input(ui, allow: bool) -> None:
   """Enable or disable user input"""
   for x in (
      #ui.checkBox,
      #ui.btn_norm,
      ui.btn_load,
      ui.tw_weights,
      #ui.cb_direct,
      ui.tw_kernel,
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
   global KERNELS
   B = Branches(9)

   for dx in range(-2,2+1):
      for dy in range(-2,2+1):
         B[dy,dx] = KERNELS[dy+2,dx+2,:,:]

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
   global SELECTED_ROW, SELECTED_COL, KERNELS
   SELECTED_ROW = 2
   SELECTED_COL = 2
   KERNELS = np.zeros((5,5,9,9))

   app = QtWidgets.QApplication(sys.argv)
   MainWindow = QtWidgets.QMainWindow()
   ui = Ui_MainWindow()
   ui.setupUi(MainWindow)

   ui.actionQuit.triggered.connect(MainWindow.close)
   ui.btn_load.clicked.connect(Handles.click_load)
   ui.btn_in2out.clicked.connect(Handles.click_compute)
   ui.btn_out2in.clicked.connect(Handles.click_copypaste)
   ui.tw_weights.selectionModel().selectionChanged.connect(Handles.select_weight)
   ui.tw_kernel.cellChanged.connect(Handles.kernel_update)

   MainWindow.show()
   sys.exit(app.exec_())
