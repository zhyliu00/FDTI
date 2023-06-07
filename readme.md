# FDTI: Fine-grained Deep Traffic Inference with Roadnet-enriched Graph

[ECML PKDD 2023]This is a Pytorch and DGL implementation of the following paper : "FDTI: Fine-grained Deep Traffic Inference with Roadnet-enriched Graph".

## Data
Please download the Manhattan data from https://drive.google.com/file/d/1TxVluhAEU3oFhlzoXq6FxmP7TTJjOuZG/view?usp=sharing and unzip it in `./data` folder

## Directory
The Root is described as below


```
ROOT
+-- data
|   +-- manhattan_train.dgl
|   +-- manhattan_val.dgl
|   +-- ...
+-- outputs
|   +-- ...
+-- model.py
+-- test.py
+-- train.py
+-- utils.py
```

- `data` the dataset folder. Here it contains Manhattan dataset.
- `outputs` contains the training log and model file. Each time run `train.py` to launch a new training process, it will automatically create a folder to store the training log and model file.
- `train.py` python script of training FDTI .
- `test.py` python script of testing the model.
- `utils.py` some useful function
- `model.py` contains the GNN module.
## For training


```commandline
python train.py --setting manhattan 
```

After finish training, a process for evaluation will be automatically created.  


## For evaluation
```commandline
python test.py --setting manhattan --test_setting manhattan --test_folder $TRAIN_FOLDER
```

*$TRAIN_FOLDER* is the result folder name in *./outputs/manhattan/* which is created based on the time that training process
begins.

