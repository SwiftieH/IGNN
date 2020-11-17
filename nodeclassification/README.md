# Node classification

## Datasets
For Amazon dataset, please use the following dropbox link to download:
```
https://www.dropbox.com/sh/3gwr2wgh455q9pi/AAB0i6EQimVGslrqtsTIWsL0a?dl=0
```
Then put the `amazon-all` folder under the `data` folder.

For PPI dataset, simply have `torch_geometric` installed and run the code.

## Run
- For the experiment on chains dataset:
```bash
python train_IGNN_chains.py
```
- For the experiment on amazon dataset:
```bash
python train_IGNN_amazon.py
```
- For the experiment on PPI dataset:
```bash
python train_IGNN_PPI.py
```
Then you should get the results in paper. To get better performance, tuning the hyper-parameters is highly encouraged.
