# Implicit Graph Neural Networks
This repository is the official PyTorch implementation of "Implicit Graph Neural Networks".

Fangda Gu*, Heng Chang*, Wenwu Zhu, Somayeh Sojoudi, Laurent El Ghaoui, [Implicit Graph Neural Networks](https://proceedings.neurips.cc/paper/2020/file/8b5c8441a8ff8e151b191c53c1842a38-Paper.pdf), NeurIPS 2020.

## Requirements
The script has been tested running under Python 3.6.9, with the following packages installed (along with their dependencies):
* pytorch (tested on 1.6.0)
* torch_geometric (tested on 1.6.1)
* scipy (tested on 1.5.2)
* numpy (tested on 1.19.2)

## Tasks
We provide examples on the tasks of node classification and graph classification consistent with the experimental results of our paper. Please refer to ``nodeclassification`` and ``graphclassification`` for usage.

## Reference
- If you find ``IGNN`` useful in your research, please cite the following in your manuscript:

```
@inproceedings{gu2020implicit,
 author = {Gu, Fangda and Chang, Heng and Zhu, Wenwu and Sojoudi, Somayeh and El Ghaoui, Laurent},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {11984--11995},
 publisher = {Curran Associates, Inc.},
 title = {Implicit Graph Neural Networks},
 volume = {33},
 year = {2020}
}

```

