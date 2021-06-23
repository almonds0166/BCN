# BCN

**Madison Landry**

This repository contains the source code for my master's thesis project, in which I investigate an abstraction of sparse neural networks called branched connection networks (BCNs) that appear in [neuromorphic hardware](https://en.wikipedia.org/wiki/Neuromorphic_engineering), particularly optics-based neuromorphic hardware, to uncover strategies to better train these networks and devices.

[Code documentation is hosted here](https://web.mit.edu/almonds/www/BCN/index.html).

<!-- ## Running locally

Follow [these instructions](https://pytorch.org/get-started/locally/) to install PyTorch locally; then install the requirements in `requirements.txt`. -->

## TODO

* Weight perturbation
* Add BCN method to precision, recall, etc. per label in addition to average?
* Connections/branches should use numpy arrays instead of torch tensors (much faster according to testing!) and 16 instead of 32 (to save disk space)
* Differentiate simple branches from uniform branches of the same name
* Come up with cleaner way to save & load weights & results
* Convert weights dict to tensor

## GUI TODO
* Investigate `QTableWidget: cannot insert an item that is already owned by another QTableWidget` issue
* Fix Vital branched connections to 9x9 instead of 7x7
* Add button to clear input/output planes

### Further along
