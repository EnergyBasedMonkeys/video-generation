# Energy Based Monkeys

## Install/Setup

To install the dependencies run
```
pip install -e .
```

Once installed you shoudl be able to run both 
```
segnet --help
transformer --help
```

## Sgenet Model
The first part of the pipeline is to train a Segnet Model, the dataset is assumed to be in a `<dir>/train` and `<dir>/val` for training and `<dir>/hidden` for inference

### Train

To train you can run

```
segnet train --output-dir ./segent-out --dataset-dir ./dataset/dir
```

### Inference

To produce a prediction file you can run

```
segnet inference --segnet-checkpoint ./segent-out/checkpoint-0.pt --dataset-dir ./dataset/dir --output-dir ./segent-out
```

## Transformer Model
Once the segnet model is trained, you can use the trained wwights to train the full e2e transformer based model.
The dataset is assumed to be in a "<dir>/train" for labeled data, "<dir>/unlabeled" for unlalabeled data and "<dir>/val" for training and "<dir>/hidden" for inference

### Train

To train you can run

```
transformer train --output-dir ./transformer-out --dataset-dir ./dataset/dir
```

### Inference

To produce a prediction file you can run

```
transformer inference --transformer-checkpoint ./transformer-out/checkpoint-0.pt --dataset-dir ./dataset/dir --output-dir ./transformer-out
```


## Running on HPC
You can see how to run each of these steps on a HPC (i.e. Greene) by looking in the slurm directory

```
sbatch segent_train.slurm
sbatch transformer_train.slurm
sbatch transformer_predict.slurm
```
