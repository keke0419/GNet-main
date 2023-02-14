# GNet-main

An integrated context-aware neural framework for transcription factor binding signal at single nucleotide resolution prediction

## Requirements

- Pytorch 1.1 [[Install]](https://pytorch.org/)
- CUDA 10.0
- Python 3.6
- Gensim 3.8.3
- Other Python packages: biopython, scikit-learn, pandas, pyBigWig, scipy, matplotlib, seaborn, nltk
- Download [hg38.fa, mm10.fa](https://hgdownload.soe.ucsc.edu/downloads.html) then unzip them and put them into `Genome/`

## Data Preparation

- Sequence preprocessing and one-hot coding:
```
python bedsignal-hg38.py -d <> -n <> -s <>
```

| Arguments   | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| -d          | The path of datasets, e.g. /your_path/GNet-main/GM12878/CTCF   |
| -n          | The name of the specified dataset, e.g. CTCF                   |
| -s          | Random seed (default is 666)                                   |

- ND and RFHC coding:

```
python enbinding.py -d <> -n <> -c <> -s <>
```
| Arguments   | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| -d          | The path of datasets, e.g. /your_path/GNet-main/GM12878/CTCF   |
| -n          | The name of the specified dataset, e.g. CTCF                   |
| -c          | One of the train, test and neg, e.g. train                     |
| -s          | Random seed (default is 666)                                   |

## Model Training

Regression and classification train for specified data sets:

```
python run.py -d <> -n <> -g <> -s <> -b <> -e <> -c <>
```

| Arguments  | Description                                                                      |
| ---------- | -------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/GNet-main/GM12878/CTCF/data     |
| -n         | The name of the specified dataset, e.g. CTCF                                     |
| -g         | The GPU device id (default is 0)                                                 |
| -s         | Random seed                                                                      |
| -b         | The number of sequences in a batch size (default is 500)                         |
| -e         | The epoch of training steps (default is 50)                                      |
| -c         | The path for storing models, e.g. /your_path/GNet-main/models/GM12878/CTCF       |

Output:

The trained model can be found in `/your_path/GNet-main/models/GM12878/CTCF/model_best.pth`. 

## Model Testing

Regression and classification tests for specified data set:

```
python test.py -d <> -n <> -g <> -c <>
```

| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/GNet-main/GM12878/CTCF/data                |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -c         | The trained model path of a specified dataset, e.g. /your_path/GNet-main/models/GM12878/CTCF|

Output:

The generated test results can be found in `/your_path/GNet-main/models/GM12878/CTCF/record.txt`. These include regression evaluation indicators MSE and PCC values, as well as classified evaluation indicators AUC and AUPRC values.

## Motif Prediction

Motif prediction for specified test data:

```
python motif.py -d <> -n <> -g <> -t <> -c <> -o <>
```

| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/GNet-main/GM12878/CTCF/data                |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -t         | The threshold value (default is 0.3)                                                        |
| -c         | The trained model path of a specified dataset, e.g. /your_path/GNet-main/models/GM12878/CTCF|
| -o         | The path of storing motif files, e.g. /your_path/GNet-main/motifs/GM12878/CTCF              |

Output:

The resulting motif file can be found in `/your_path/GNet-main/motifs/GM12878/CTCF/motif.meme` and applied to the TOMTOM motif comparison tool.
