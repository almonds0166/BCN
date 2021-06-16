# BCN

**Madison Landry**

This repository contains the source code for my master's thesis project, in which I investigate an abstraction of sparse neural networks called branched connection networks (BCNs) that appear in [neuromorphic hardware](https://en.wikipedia.org/wiki/Neuromorphic_engineering), particularly optics-based neuromorphic hardware, to uncover strategies to better train these networks and devices.

<!-- ## Running locally

Follow [these instructions](https://pytorch.org/get-started/locally/) to install PyTorch locally; then install the requirements in `requirements.txt`. -->

## TODO

* Improve GUI (9x9 instead of 5x5)
* Weight perturbation
* Add BCN method to precision, recall, etc. per label in addition to average?
* Connections/branches should use numpy arrays instead of torch tensors (much faster according to testing!) and 16 instead of 32 (to save disk space)
* Differentiate simple branches from uniform branches of the same name
* Come up with cleaner way to save & load weights & results

### Further along
