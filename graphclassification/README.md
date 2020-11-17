# Graph classification

## Datasets

For all datasets used for graph classification, simply have `torch_geometric` installed and run the code.

## Run
- The experiments are self-contained, where you need to specify the name of dataset and then execute the script. For example:
```bash
python train_IGNN.py --dataset MUTAG
```

Then you should get the results in paper. To get better performance, tuning the hyper-parameters is highly encouraged.
