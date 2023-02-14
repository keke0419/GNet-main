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

Sequence preprocessing and one-hot coding:
```
python bedsignal-hg38.py -d <> -n <> -s <>
```

| Arguments   | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| -d          | The path of datasets, e.g. /your_path/GNet-main/GM12878/CTCF   |
| -n          | The name of the specified dataset, e.g. CTCF                   |
| -s          | Random seed (default is 666)                                   |

ND and RFHC coding:

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

